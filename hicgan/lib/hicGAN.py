import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU, Flatten, Dense
from tensorflow.keras.layers import GroupNormalization
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import utils
import re

# Conditional DDPM / DDIM Diffusion model sharing the Phase 1 1D->2D chromatin condition encoder
# and U-Net backbone for fair comparison against Pix2Pix cGAN (rna-cgan-clean).

import logging
log = logging.getLogger(__name__)


def build_condition_encoder(
    input_size: int,
    number_factors: int,
    use_pair_features: bool = False,
    enforce_symmetry: bool = False,
    nr_filters_list=[128, 128, 64, 32],
    kernel_width_list=[5, 5, 5, 5],
    apply_dropout: bool = False,
    name: str = "CNN-embedding",
):
    """
    Modular 1D-to-2D Chromatin Condition Encoder with 2D Genomic Distance Encoding (reusable across cGAN and Diffusion):
      1. Splits 1D input into row and column chromatin streams (+ genomic coordinate channels in 'grid' mode).
      2. Encodes row and column 1D spans (length 3*W) with a shared multi-scale dilated Conv1D trunk.
      3. Extracts the exact center window [W : 2*W] aligned 1-to-1 with the WxW target bins alongside
         global strided context, and forms a 2D pairwise interaction tensor at every (i, j) pixel:
           [E_row * E_col, (E_row * E_col) * d_inv, (E_row * E_col) * pair_2d, 0.5*(E_row + E_col), |E_row - E_col|, E_row, E_col, D(i, j)]
         where D(i, j) encodes the exact genomic distance |(r0 + i) - (c0 + j)| via 4 multi-scale decay maps (+ optional 2D pair_features).
      4. Refines the 2D representation with Conv2D layers and outputs a (W, W, 16) 2D feature + distance tensor `x_emb`.
    """
    inputs = tf.keras.layers.Input(shape=(3 * input_size, number_factors))
    has_pair_2d = bool(use_pair_features) and (number_factors > input_size)
    n_1d_total = (number_factors - input_size) if has_pair_2d else number_factors
    is_grid_mode = (not enforce_symmetry) and (n_1d_total >= 4) and (n_1d_total % 2 == 0)
    n_chrom_factors = (n_1d_total // 2) - 1 if is_grid_mode else n_1d_total

    f_row, f_col, pos_row, pos_col, pair_mat_2d = SplitChromatinStreamsLayer(
        window_size=input_size,
        number_factors=number_factors,
        is_grid_mode=is_grid_mode,
        has_pair_2d=has_pair_2d,
        name="split_chromatin_streams",
    )(inputs)

    # Shared Siamese 1D multi-scale dilated encoder for 1D chromatin tracks
    seq_in = tf.keras.layers.Input(shape=(3 * input_size, n_chrom_factors), name="chrom_1d_in")
    h = seq_in
    dilation_rates = [1, 2, 4, 8]
    for i, (nr_filters, kernelWidth, dil) in enumerate(zip(nr_filters_list, kernel_width_list, dilation_rates)):
        h = Conv1D(
            filters=nr_filters,
            kernel_size=kernelWidth,
            dilation_rate=dil,
            padding="same",
            data_format="channels_last",
            name=f"conv1D_{i + 1}",
        )(h)
        n_groups = min(8, max(1, nr_filters // 4))
        h = GroupNormalization(groups=n_groups, axis=-1, name=f"gn1D_{i + 1}")(h)
        if apply_dropout:
            h = Dropout(0.3, name=f"drop1D_{i + 1}")(h)
        h = LeakyReLU(alpha=0.2, name=f"act1D_{i + 1}")(h)

    h_stride = Conv1D(
        filters=16,
        strides=3,
        kernel_size=5,
        padding="same",
        data_format="channels_last",
        name="conv1D_stride_context",
    )(h)
    h_stride = LeakyReLU(alpha=0.2, name="act1D_stride_context")(h_stride)
    chrom_1d_encoder = tf.keras.Model(inputs=seq_in, outputs=[h, h_stride], name="Chromatin1DEncoder")

    h_row_full, h_row_stride = chrom_1d_encoder(f_row)
    if is_grid_mode:
        h_col_full, h_col_stride = chrom_1d_encoder(f_col)
    else:
        h_col_full, h_col_stride = h_row_full, h_row_stride

    pair_2d, dist_2d = Pairwise2DEmbeddingLayer(
        window_size=input_size,
        is_grid_mode=is_grid_mode,
        has_pair_2d=has_pair_2d,
        name="pairwise_2d_embedding",
    )([f_row, f_col, h_row_full, h_col_full, h_row_stride, h_col_stride, pos_row, pos_col, pair_mat_2d])

    initializer = tf.random_normal_initializer(0.0, 0.02)
    z = Conv2D(64, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, name="emb_conv2d_1")(pair_2d)
    z = GroupNormalization(groups=8, axis=-1, name="emb_gn2d_1")(z)
    z = LeakyReLU(alpha=0.2, name="emb_act2d_1")(z)
    n_emb_conv2 = 11 if has_pair_2d else 12
    z = Conv2D(n_emb_conv2, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, name="emb_conv2d_2")(z)
    z = LeakyReLU(alpha=0.2, name="emb_act2d_2")(z)

    x = tf.keras.layers.Concatenate(axis=-1, name="emb_concat_dist")([z, dist_2d])
    if enforce_symmetry:
        x = Symmetrize2DLayer(name="emb_symmetrize")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    return model


@tf.keras.utils.register_keras_serializable(package="hicgan")
class SpatialTimeEncoding(tf.keras.layers.Layer):
    """Broadcast normalized diffusion timestep t / T into a (batch, W, W, 1) spatial map."""
    def __init__(self, timesteps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = float(timesteps)

    def call(self, inputs):
        ref_tensor, t = inputs
        t_scaled = tf.cast(t, ref_tensor.dtype) / tf.cast(self.timesteps, ref_tensor.dtype)
        t_expanded = tf.reshape(t_scaled, [-1, 1, 1, 1])
        ref_shape = tf.shape(ref_tensor)
        return tf.broadcast_to(t_expanded, [ref_shape[0], ref_shape[1], ref_shape[2], 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"timesteps": self.timesteps})
        return cfg


class HiCGAN():
    def __init__(self, log_dir: str,
                    number_factors: int,
                    timesteps: int = 1000,
                    sample_steps: int = 25,
                    loss_weight_pixel: float = 1.0,
                    foreground_loss_weight: float = 15.0,
                    peak_loss_weight: float = 10.0,
                    x0_loss_weight: float = 1.0,
                    bg_l1_weight: float = 0.5,
                    leaky_clip_slope: float = 0.25,
                    dropout_rate: float = 0.0,
                    decay_steps: int = None,
                    lr_alpha: float = 0.05,
                    input_size: int = 256,
                    plot_frequency: int = 20,
                    plot_type: str = "png",
                    learning_rate: float = 2e-5,
                    learning_rate_generator: float = None,
                    adam_beta_1: float = 0.5,
                    pretrained_model_path: str = "",
                    scope=None,
                    mixed_precision: bool = False,
                    restore_checkpoint: bool = False,
                    use_embedding: bool = True,
                    enforce_symmetry: bool = True,
                    use_pair_features: bool = False,
                    loss_mode: str = "current",
                    raw_target_mode: bool = False):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.use_embedding = bool(use_embedding)
        self.enforce_symmetry = bool(enforce_symmetry)
        self.loss_mode = str(loss_mode or "current")
        self.raw_target_mode = bool(raw_target_mode)

        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.EMBEDDING_CHANNELS = 16
        self.input_size = 256
        if input_size in [64, 128, 256, 512, 768, 1024]:
            self.input_size = input_size
        self.number_factors = number_factors
        self.use_pair_features = bool(use_pair_features) and (self.number_factors > self.input_size)

        # Forward DDPM diffusion schedule: beta_0 = 0.0 (alpha_bar_0 = 1.0), linear betas for t = 1..T
        self.timesteps = max(1, int(timesteps))
        self.sample_steps = max(1, int(sample_steps))
        beta_start = 1e-4
        beta_end = 0.02
        betas_steps = tf.cast(tf.linspace(beta_start, beta_end, self.timesteps), tf.float32)
        self.betas = tf.concat([[0.0], betas_steps], axis=0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = tf.math.cumprod(self.alphas)

        self.loss_weight_pixel = float(loss_weight_pixel)
        self.fg_weight = float(foreground_loss_weight)
        self.peak_weight = float(peak_loss_weight)
        self.x0_loss_weight = float(x0_loss_weight)
        self.bg_l1_weight = float(bg_l1_weight)
        self.leaky_clip_slope = float(leaky_clip_slope)
        self.dropout_rate = float(dropout_rate)
        self.decay_steps = int(decay_steps) if (decay_steps is not None and int(decay_steps) > 0) else None
        self.lr_alpha = float(lr_alpha)

        eff_lr = float(learning_rate_generator if learning_rate_generator is not None else learning_rate)

        init_msg = (
            f"[MODEL-INIT] HiCGAN-Diffusion configuration: input_size={self.input_size}, number_factors={self.number_factors}, "
            f"timesteps={self.timesteps}, sample_steps={self.sample_steps}, "
            f"use_embedding={self.use_embedding} ({'1D -> Pairwise 2D + Distance cnn_embedding' if self.use_embedding else 'Direct 2D Input'}), "
            f"use_pair_features={self.use_pair_features}, enforce_symmetry={self.enforce_symmetry}, "
            f"loss_mode={self.loss_mode}, raw_target_mode={self.raw_target_mode}, "
            f"fg_weight={self.fg_weight}, peak_weight={self.peak_weight}, x0_loss_weight={self.x0_loss_weight}, "
            f"bg_l1_weight={self.bg_l1_weight}, leaky_clip_slope={self.leaky_clip_slope}, "
            f"dropout_rate={self.dropout_rate}, lr={eff_lr:.2e}, decay_steps={self.decay_steps}, mixed_precision={self.mixed_precision}"
        )
        log.info(init_msg)
        print(init_msg)

        if self.decay_steps is not None:
            opt_lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=eff_lr,
                decay_steps=self.decay_steps,
                alpha=self.lr_alpha,
            )
        else:
            opt_lr = eff_lr

        self.denoiser_optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr, beta_1=adam_beta_1, name="Adam_Denoiser")
        if self.mixed_precision:
            self.denoiser_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.denoiser_optimizer)

        if self.use_embedding:
            self.condition_encoder = self.build_condition_encoder()
            cond_channels = self.EMBEDDING_CHANNELS
        else:
            self.condition_encoder = None
            cond_channels = self.INPUT_CHANNELS

        self.unet_denoiser = self._build_unet_denoiser(cond_channels=cond_channels)
        self.denoiser = self.Denoiser()

        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            denoiser_optimizer=self.denoiser_optimizer,
            denoiser=self.denoiser,
            ckpt_epoch=tf.Variable(0, trainable=False, dtype=tf.int64, name="ckpt_epoch"),
        )
        self.ckpt_epoch = self.checkpoint.ckpt_epoch
        self.start_epoch = 0
        if restore_checkpoint:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            print(f"Current checkpoint directory: {self.checkpoint_dir}")
            if latest_checkpoint:
                print(f"Restoring from checkpoint: {latest_checkpoint}")
                self.checkpoint.restore(latest_checkpoint).expect_partial()
                stored_ep = int(self.ckpt_epoch.numpy())
                if stored_ep > 0:
                    ckpt_num = stored_ep
                else:
                    basename = os.path.basename(latest_checkpoint)
                    m = re.search(r'ckpt-(\d+)$', basename)
                    if m:
                        ckpt_num = int(m.group(1))
                    else:
                        try:
                            ckpt_num = int(basename.split('-')[-1])
                        except Exception:
                            ckpt_num = 0
                self.start_epoch = ckpt_num
                print(f"Resuming from checkpoint epoch #{ckpt_num}.")
            else:
                print("No checkpoint found. Starting from scratch.")

        self.plot_type = plot_type
        if self.plot_type not in ["png", "pdf", "svg"]:
            self.plot_type = "png"
            print("plot type {:s} unsupported, changed to png".format(plot_type))
        self.progress_plot_name = os.path.join(self.log_dir, "lossOverEpochs.{:s}".format(self.plot_type))
        self.progress_plot_frequency = plot_frequency
        self.example_plot_frequency = 5

        # Losses per epoch
        self.__diff_train_loss_epochs = []
        self.__diff_train_noise_loss_epochs = []
        self.__diff_train_x0_loss_epochs = []
        self.__diff_train_bg_loss_epochs = []
        self.__diff_val_loss_epochs = []
        self.__diff_val_noise_loss_epochs = []
        self.__diff_val_x0_loss_epochs = []
        self.__diff_val_bg_loss_epochs = []

        # Losses per batch
        self.__diff_train_loss_batches = []
        self.__diff_train_noise_loss_batches = []
        self.__diff_train_x0_loss_batches = []
        self.__diff_train_bg_loss_batches = []
        self.__diff_val_loss_batches = []
        self.__diff_val_noise_loss_batches = []
        self.__diff_val_x0_loss_batches = []
        self.__diff_val_bg_loss_batches = []

        self.__epoch_counter = 0
        self.__batch_counter = 0

        if scope is None:
            scope = tf.distribute.get_strategy()
        self.scope = scope

    def build_condition_encoder(self, nr_filters_list=[128, 128, 64, 32], kernel_width_list=[5, 5, 5, 5], apply_dropout: bool = False):
        return build_condition_encoder(
            input_size=self.input_size,
            number_factors=self.number_factors,
            use_pair_features=self.use_pair_features,
            enforce_symmetry=self.enforce_symmetry,
            nr_filters_list=nr_filters_list,
            kernel_width_list=kernel_width_list,
            apply_dropout=apply_dropout,
            name="CNN-embedding",
        )

    def cnn_embedding(self, nr_filters_list=[128, 128, 64, 32], kernel_width_list=[5, 5, 5, 5], apply_dropout: bool = False):
        return self.build_condition_encoder(
            nr_filters_list=nr_filters_list,
            kernel_width_list=kernel_width_list,
            apply_dropout=apply_dropout,
        )

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            n_groups = min(32, max(1, filters // 8))
            result.add(GroupNormalization(groups=n_groups, axis=-1))
        result.add(LeakyReLU(alpha=0.2))
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False, dropout_rate: float = 0.0):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   use_bias=False))
        n_groups = min(32, max(1, filters // 8))
        result.add(GroupNormalization(groups=n_groups, axis=-1))
        if apply_dropout and dropout_rate > 0.0:
            result.add(Dropout(float(dropout_rate)))
        result.add(ReLU())
        return result

    def _build_unet_denoiser(self, cond_channels: int):
        """
        Core U-Net denoiser operating on [noisy_target (W,W,1), cond_emb (W,W,C_emb), time_step (1,)].
        Matches Phase 1 Generator U-Net depth, widths, GroupNormalization, and Dropout exactly,
        replacing the final Symmetrize2DLayer + sigmoid head with unconstrained linear noise prediction.
        Separating this sub-model allows `condition_encoder` to be computed ONCE and cached across
        all 25 DDIM reverse diffusion steps during inference.
        """
        noisy_target = tf.keras.layers.Input(shape=[self.input_size, self.input_size, 1], name="noisy_target_core")
        cond_emb = tf.keras.layers.Input(shape=[self.input_size, self.input_size, cond_channels], name="cond_emb_core")
        time_step = tf.keras.layers.Input(shape=[1], name="time_step_core")

        time_map = SpatialTimeEncoding(timesteps=self.timesteps, name="spatial_time_encoding")([noisy_target, time_step])
        x = tf.keras.layers.Concatenate(axis=-1, name="diff_concat_inputs")([noisy_target, cond_emb, time_map])

        num_downsamples = int(np.round(np.log2(self.input_size)))
        down_filters = [min(64 * (2 ** i), 512) for i in range(num_downsamples)]
        down_stack = [
            HiCGAN.downsample(
                f, 4, apply_batchnorm=(i != 0 and i != num_downsamples - 1)
            )
            for i, f in enumerate(down_filters)
        ]

        up_filters = [min(64 * (2 ** i), 512) for i in reversed(range(num_downsamples - 1))]
        up_stack = [
            HiCGAN.upsample(
                f, 4,
                apply_dropout=(i < 3 and self.dropout_rate > 0.0),
                dropout_rate=self.dropout_rate,
            )
            for i, f in enumerate(up_filters)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        # Upsample to full (W, W) resolution and fuse with full-resolution condition embedding `cond_emb`
        x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
        x = GroupNormalization(groups=8, axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Concatenate()([x, cond_emb])
        x = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # Unconstrained linear noise prediction epsilon_hat (no Symmetrize2DLayer, no sigmoid)
        x = Conv2D(self.OUTPUT_CHANNELS, 1, strides=1, padding='same', kernel_initializer=initializer)(x)

        return tf.keras.Model(inputs=[noisy_target, cond_emb, time_step], outputs=x, name="UNet_Diffusion_Core")

    def Denoiser(self):
        noisy_target = tf.keras.layers.Input(shape=[self.input_size, self.input_size, 1], name="noisy_target")
        if self.use_embedding:
            factor_input = tf.keras.layers.Input(shape=[3 * self.input_size, self.number_factors], name="factorData")
            cond_emb = self.condition_encoder(factor_input)
        else:
            factor_input = tf.keras.layers.Input(shape=[self.input_size, self.input_size, self.INPUT_CHANNELS], name="factorData")
            cond_emb = factor_input
        time_step = tf.keras.layers.Input(shape=[1], name="time_step")
        pred_noise = self.unet_denoiser([noisy_target, cond_emb, time_step])
        return tf.keras.Model(inputs=[noisy_target, factor_input, time_step], outputs=pred_noise, name="UNet_Diffusion")

    def diffusion_loss(self, target, noise, predicted_noise, noisy_target, alpha_bar_t):
        target = tf.cast(target, tf.float32)
        noise = tf.cast(noise, tf.float32)
        predicted_noise = tf.cast(predicted_noise, tf.float32)
        noisy_target = tf.cast(noisy_target, tf.float32)
        alpha_bar_t = tf.cast(alpha_bar_t, tf.float32)

        fg_mask = tf.cast(target > 1e-5, tf.float32)
        pixel_weights = (
            1.0
            + self.fg_weight * fg_mask
            + self.peak_weight * tf.clip_by_value(target, 0.0, 1.0)
        )
        noise_loss = tf.reduce_mean(pixel_weights * tf.square(noise - predicted_noise))

        sqrt_ab = tf.sqrt(tf.maximum(alpha_bar_t, 1e-8))
        sqrt_one_minus_ab = tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-8))
        x0_pred_raw = (noisy_target - sqrt_one_minus_ab * predicted_noise) / sqrt_ab
        if self.raw_target_mode:
            x0_pred_clipped = tf.maximum(x0_pred_raw, 0.0)
        else:
            x0_pred_clipped = tf.clip_by_value(x0_pred_raw, 0.0, 1.0)
        x0_pred = x0_pred_clipped + self.leaky_clip_slope * (x0_pred_raw - tf.stop_gradient(x0_pred_clipped))
        x0_loss = tf.reduce_mean(alpha_bar_t * pixel_weights * tf.square(target - x0_pred))

        zero_mask = 1.0 - fg_mask
        bg_l1_loss = tf.reduce_mean(alpha_bar_t * zero_mask * tf.abs(x0_pred))

        total_loss = self.loss_weight_pixel * (
            noise_loss
            + self.x0_loss_weight * x0_loss
            + self.bg_l1_weight * bg_l1_loss
        )
        return total_loss, noise_loss, x0_loss, bg_l1_loss

    @tf.function
    def distributed_train_step(self, input_data):
        input_image, target = input_data[0]["factorData"], input_data[1]["out_matrixData"]
        per_replica_losses = self.scope.run(self.train_step, args=(input_image, target, ))
        return self.scope.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    @tf.function
    def train_step(self, input_image, target):
        target = tf.cast(target, tf.float32)
        batch_size = tf.shape(target)[0]

        t = tf.random.uniform(shape=[batch_size, 1], minval=1, maxval=self.timesteps + 1, dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(target), dtype=tf.float32)

        alpha_bar_t = tf.gather(self.alphas_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])
        noisy_target = tf.sqrt(alpha_bar_t) * target + tf.sqrt(1.0 - alpha_bar_t) * noise

        with tf.GradientTape() as tape:
            predicted_noise = self.denoiser([noisy_target, input_image, tf.cast(t, tf.float32)], training=True)
            total_loss, noise_loss, x0_loss, bg_l1_loss = self.diffusion_loss(
                target, noise, predicted_noise, noisy_target, alpha_bar_t
            )
            if self.mixed_precision:
                if hasattr(self.denoiser_optimizer, "get_scaled_loss"):
                    scaled_loss = self.denoiser_optimizer.get_scaled_loss(total_loss)
                elif hasattr(self.denoiser_optimizer, "scale_loss"):
                    scaled_loss = self.denoiser_optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss

        if self.mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.denoiser.trainable_variables)
            if hasattr(self.denoiser_optimizer, "get_unscaled_gradients"):
                gradients = self.denoiser_optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = scaled_gradients
        else:
            gradients = tape.gradient(total_loss, self.denoiser.trainable_variables)

        self.denoiser_optimizer.apply_gradients(zip(gradients, self.denoiser.trainable_variables))
        return total_loss, noise_loss, x0_loss, bg_l1_loss

    @tf.function
    def validationStep(self, input_image, target):
        target = tf.cast(target, tf.float32)
        batch_size = tf.shape(target)[0]

        # Deterministic stratified timesteps + stateless noise for noise-free EarlyStopping monitoring
        t_lin = tf.linspace(1.0, tf.cast(self.timesteps, tf.float32), tf.maximum(batch_size, 1))
        t = tf.reshape(tf.cast(tf.round(t_lin), tf.int32)[:batch_size], [batch_size, 1])
        noise = tf.random.stateless_normal(shape=tf.shape(target), seed=[42, 42], dtype=tf.float32)

        alpha_bar_t = tf.gather(self.alphas_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])
        noisy_target = tf.sqrt(alpha_bar_t) * target + tf.sqrt(1.0 - alpha_bar_t) * noise

        predicted_noise = self.denoiser([noisy_target, input_image, tf.cast(t, tf.float32)], training=False)
        return self.diffusion_loss(target, noise, predicted_noise, noisy_target, alpha_bar_t)

    @tf.function
    def predictionStep(self, input_batch, training=False, sample_steps: int = None):
        """
        Deterministic DDIM reverse sampling (default 25 steps, seed=[42, 42]) with [0, 1]-clipped x_0
        effective noise correction. Caches `self.condition_encoder(input_batch)` ONCE per batch
        across all DDIM steps.
        """
        steps = self.sample_steps if sample_steps is None else int(sample_steps)
        inp = tf.cast(input_batch, tf.float32)
        if not self.use_embedding and inp.shape.rank == 3:
            inp = tf.expand_dims(inp, axis=-1)
        batch_size = tf.shape(inp)[0]

        if self.use_embedding and self.condition_encoder is not None:
            cond_emb = self.condition_encoder(inp, training=False)
        else:
            cond_emb = inp

        x_t = tf.random.stateless_normal(
            shape=(batch_size, self.input_size, self.input_size, 1),
            seed=[42, 42],
            dtype=tf.float32,
        )
        stride = tf.maximum(1, self.timesteps // tf.maximum(steps, 1))

        for t_step in tf.range(self.timesteps, 0, -stride):
            t_batch = tf.fill([batch_size, 1], tf.cast(t_step, tf.float32))
            t_prev = tf.maximum(0, t_step - stride)

            predicted_noise = tf.cast(
                self.unet_denoiser([x_t, cond_emb, t_batch], training=False),
                tf.float32,
            )

            alpha_bar_t = self.alphas_bar[t_step]
            alpha_bar_prev = self.alphas_bar[t_prev]

            sqrt_ab_t = tf.sqrt(tf.maximum(alpha_bar_t, 1e-8))
            sqrt_one_minus_ab_t = tf.sqrt(tf.maximum(1.0 - alpha_bar_t, 1e-8))

            x0_pred = (x_t - sqrt_one_minus_ab_t * predicted_noise) / sqrt_ab_t
            if self.raw_target_mode:
                x0_pred = tf.maximum(x0_pred, 0.0)
            else:
                x0_pred = tf.clip_by_value(x0_pred, 0.0, 1.0)
            pred_eps = (x_t - sqrt_ab_t * x0_pred) / sqrt_one_minus_ab_t
            x_t = (
                tf.sqrt(alpha_bar_prev) * x0_pred
                + tf.sqrt(tf.maximum(1.0 - alpha_bar_prev, 0.0)) * pred_eps
            )

        if self.raw_target_mode:
            x_t = tf.maximum(x_t, 0.0)
        else:
            x_t = tf.clip_by_value(x_t, 0.0, 1.0)
        if self.enforce_symmetry:
            x_t = 0.5 * (x_t + tf.transpose(x_t, perm=[0, 2, 1, 3]))
        return x_t

    def generate_images(self, model, test_input, tar, epoch: int):
        from . import evaluation

        inp_tensor = test_input["factorData"] if isinstance(test_input, dict) else test_input
        prediction = self.predictionStep(inp_tensor[:1], training=False)
        tar_img = tf.cast(tar["out_matrixData"][0], tf.float32)
        pred_img = tf.cast(prediction[0], tf.float32)

        tar_raw = np.squeeze(np.asarray(tar_img, dtype=np.float64))
        pred_raw = np.squeeze(np.asarray(pred_img, dtype=np.float64))
        pred_mse = float(np.mean(np.square(tar_raw - pred_raw)))
        if self.raw_target_mode:
            gt_disp = np.maximum(tar_raw, 0.0)
            pred_disp = np.maximum(pred_raw, 0.0)
        else:
            gt_disp = np.clip(tar_raw, 0.0, 1.0)
            pred_disp = np.clip(pred_raw, 0.0, 1.0)
        mae_val = float(np.mean(np.abs(pred_disp - gt_disp)))

        inp_sample = np.asarray(inp_tensor[0], dtype=np.float64)
        inp_title = "Girdi (Input)"
        inp_xlabel = "Genomik Bin (j)"
        inp_ylabel = "Genomik Bin (i)"
        if inp_sample.ndim == 3 and inp_sample.shape[-1] == 1:
            inp_sample = inp_sample[:, :, 0]
        if (
            inp_sample.ndim == 2
            and self.use_embedding
            and self.use_pair_features
            and inp_sample.shape[0] == 3 * self.input_size
            and inp_sample.shape[1] >= self.input_size
        ):
            inp_disp = inp_sample[self.input_size : 2 * self.input_size, -self.input_size :]
            inp_title = "Girdi (2D Pair Window)"
        elif (
            inp_sample.ndim == 2
            and self.use_embedding
            and inp_sample.shape[0] == 3 * self.input_size
        ):
            inp_disp = inp_sample[self.input_size : 2 * self.input_size, :]
            inp_title = f"Girdi (1D Features, {inp_sample.shape[1]} ch)"
            inp_xlabel = "Feature Kanalı"
        elif inp_sample.ndim == 2:
            inp_disp = inp_sample
            inp_title = "Girdi (Input 2D)"
        else:
            inp_disp = np.squeeze(inp_sample)
            if inp_disp.ndim > 2:
                inp_disp = np.mean(inp_disp, axis=-1)
            inp_xlabel = "Kanal / Bin"

        figname = os.path.join(self.log_dir, "testpred_epoch_{:05d}.png".format(epoch))
        evaluation.render_comparison_panels(
            gt_mat=gt_disp,
            pred_mat=pred_disp,
            out_path=figname,
            input_mat=inp_disp,
            input_title=inp_title,
            gt_title="Gerçek (Ground Truth) [norm]",
            pred_title=f"Diffusion Tahmin (DDIM {self.sample_steps}s) [MSE={pred_mse:.5f}]",
            residual_title=f"Fark / Residual (Pred - GT) [MAE={mae_val:.4f}]",
            suptitle=f"Epoch {epoch:05d} — Diffusion DDIM Örnek Girdi, Gerçek, Tahmin ve Fark Haritası",
            input_xlabel=inp_xlabel,
            input_ylabel=inp_ylabel,
            dpi=150,
        )

    def fit(self, train_ds, epochs, test_ds, steps_per_epoch: int,
            validation_steps: int = 0, early_stopping_patience: int = 10):
        """
        Train the conditional DDPM denoiser, monitoring deterministic `val_total_loss`
        for EarlyStopping (`patience=early_stopping_patience`, `min_delta=1e-6`),
        saving `denoiser_best.keras` on improvement and restoring the best weights
        before saving `denoiser_final.keras`.
        """
        distributed_dataset = self.scope.experimental_distribute_dataset(train_ds)

        if self.start_epoch > 0 and self.start_epoch < epochs:
            print(f"Resuming training from epoch {self.start_epoch}.")
        elif self.start_epoch >= epochs:
            print(f"Checkpoint epoch {self.start_epoch} is greater than or equal to total epochs {epochs}. Nothing to train.")
            return

        vis_sample_input = None
        vis_sample_target = None
        for test_in, test_tar in test_ds:
            targets = test_tar["out_matrixData"]
            sums = tf.reduce_sum(tf.abs(targets), axis=[1, 2, 3])
            max_idx = int(tf.argmax(sums))
            if float(sums[max_idx]) > 0.0:
                vis_sample_input = {k: v[max_idx:max_idx + 1] for k, v in test_in.items()}
                vis_sample_target = {k: v[max_idx:max_idx + 1] for k, v in test_tar.items()}
                break
        if vis_sample_input is None:
            for test_in, test_tar in test_ds.take(1):
                vis_sample_input = {k: v[:1] for k, v in test_in.items()}
                vis_sample_target = {k: v[:1] for k, v in test_tar.items()}

        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        best_weights = None
        last_epoch = self.start_epoch

        def _save_loss_files(ep_idx):
            np.savez(
                os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(ep_idx)),
                diffLossTrain=self.__diff_train_loss_batches,
                diffLossVal=self.__diff_val_loss_batches,
                diffLossTrainEpochs=self.__diff_train_loss_epochs,
                diffNoiseLossTrainEpochs=self.__diff_train_noise_loss_epochs,
                diffX0LossTrainEpochs=self.__diff_train_x0_loss_epochs,
                diffBgLossTrainEpochs=self.__diff_train_bg_loss_epochs,
                diffLossValEpochs=self.__diff_val_loss_epochs,
                diffNoiseLossValEpochs=self.__diff_val_noise_loss_epochs,
                diffX0LossValEpochs=self.__diff_val_x0_loss_epochs,
                diffBgLossValEpochs=self.__diff_val_bg_loss_epochs,
                # Compatible aliases for shared tools
                genLossTrain=self.__diff_train_loss_batches,
                genLossVal=self.__diff_val_loss_batches,
                genLossTrainEpochs=self.__diff_train_loss_epochs,
                genLossValEpochs=self.__diff_val_loss_epochs,
            )

        for epoch in range(self.start_epoch, epochs):
            last_epoch = epoch
            if epoch % self.example_plot_frequency == 0 and vis_sample_input is not None:
                self.generate_images(self.denoiser, vis_sample_input, vis_sample_target, epoch)

            train_pbar = tqdm(total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch + 1))
            train_samples_in_epoch = 0

            for distributed_input in distributed_dataset:
                train_samples_in_epoch += 1
                total_l, noise_l, x0_l, bg_l = self.distributed_train_step(distributed_input)

                self.__diff_train_loss_batches.append(float(total_l))
                self.__diff_train_noise_loss_batches.append(float(noise_l))
                self.__diff_train_x0_loss_batches.append(float(x0_l))
                self.__diff_train_bg_loss_batches.append(float(bg_l))

                running_mean = float(np.mean(self.__diff_train_loss_batches[-train_samples_in_epoch:]))
                train_bar_postfixDict = {
                    "loss": "{:.4f}".format(running_mean),
                    "eps": "{:.3f}".format(self.__diff_train_noise_loss_batches[-1]),
                    "x0": "{:.3f}".format(self.__diff_train_x0_loss_batches[-1]),
                }
                if len(self.__diff_val_loss_epochs) > 0:
                    train_bar_postfixDict["val"] = "{:.3f}".format(self.__diff_val_loss_epochs[-1])
                train_pbar.set_postfix(train_bar_postfixDict)
                train_pbar.update(1)
                self.__batch_counter += 1
            train_pbar.close()

            if train_samples_in_epoch > 0:
                self.__diff_train_loss_epochs.append(float(np.mean(self.__diff_train_loss_batches[-train_samples_in_epoch:])))
                self.__diff_train_noise_loss_epochs.append(float(np.mean(self.__diff_train_noise_loss_batches[-train_samples_in_epoch:])))
                self.__diff_train_x0_loss_epochs.append(float(np.mean(self.__diff_train_x0_loss_batches[-train_samples_in_epoch:])))
                self.__diff_train_bg_loss_epochs.append(float(np.mean(self.__diff_train_bg_loss_batches[-train_samples_in_epoch:])))
            else:
                self.__diff_train_loss_epochs.append(float("nan"))
                self.__diff_train_noise_loss_epochs.append(float("nan"))
                self.__diff_train_x0_loss_epochs.append(float("nan"))
                self.__diff_train_bg_loss_epochs.append(float("nan"))

            # Deterministic validation
            validation_samples_in_epoch = 0
            vds = test_ds.take(validation_steps) if validation_steps else test_ds
            for input_image, target in vds:
                val_tot, val_noise, val_x0, val_bg = self.validationStep(input_image["factorData"], target["out_matrixData"])
                self.__diff_val_loss_batches.append(float(val_tot))
                self.__diff_val_noise_loss_batches.append(float(val_noise))
                self.__diff_val_x0_loss_batches.append(float(val_x0))
                self.__diff_val_bg_loss_batches.append(float(val_bg))
                validation_samples_in_epoch += 1

            if validation_samples_in_epoch > 0:
                self.__diff_val_loss_epochs.append(float(np.mean(self.__diff_val_loss_batches[-validation_samples_in_epoch:])))
                self.__diff_val_noise_loss_epochs.append(float(np.mean(self.__diff_val_noise_loss_batches[-validation_samples_in_epoch:])))
                self.__diff_val_x0_loss_epochs.append(float(np.mean(self.__diff_val_x0_loss_batches[-validation_samples_in_epoch:])))
                self.__diff_val_bg_loss_epochs.append(float(np.mean(self.__diff_val_bg_loss_batches[-validation_samples_in_epoch:])))
            else:
                self.__diff_val_loss_epochs.append(self.__diff_train_loss_epochs[-1])
                self.__diff_val_noise_loss_epochs.append(self.__diff_train_noise_loss_epochs[-1])
                self.__diff_val_x0_loss_epochs.append(self.__diff_train_x0_loss_epochs[-1])
                self.__diff_val_bg_loss_epochs.append(self.__diff_train_bg_loss_epochs[-1])

            val_tot_ep = self.__diff_val_loss_epochs[-1]
            log.info(
                "[EPOCH-LOSS] epoch=%d/%d | train_total=%.6f, train_noise=%.6f, train_x0=%.6f, train_bg=%.6f | "
                "val_total=%.6f, val_noise=%.6f, val_x0=%.6f, val_bg=%.6f",
                epoch + 1, epochs,
                self.__diff_train_loss_epochs[-1], self.__diff_train_noise_loss_epochs[-1],
                self.__diff_train_x0_loss_epochs[-1], self.__diff_train_bg_loss_epochs[-1],
                val_tot_ep, self.__diff_val_noise_loss_epochs[-1],
                self.__diff_val_x0_loss_epochs[-1], self.__diff_val_bg_loss_epochs[-1],
            )

            self.__epoch_counter += 1

            # EarlyStopping & best checkpoint tracking on deterministic val_total_loss
            if val_tot_ep < best_val_loss - 1e-6:
                best_val_loss = val_tot_ep
                best_epoch = epoch + 1
                patience_counter = 0
                best_weights = self.denoiser.get_weights()
                best_path = os.path.join(self.log_dir, "denoiser_best.keras")
                self.denoiser.save(filepath=best_path, save_format="keras")
                msg_best = f"[CHECKPOINT] Epoch {epoch + 1}: val_total_loss improved to {best_val_loss:.6f} -> saved {best_path}"
                log.info(msg_best)
                print(msg_best)
            else:
                patience_counter += 1
                msg_pat = (
                    f"[EARLY-STOPPING] Epoch {epoch + 1}: val_total_loss={val_tot_ep:.6f} did not improve over "
                    f"best={best_val_loss:.6f} (epoch {best_epoch}); patience={patience_counter}/{early_stopping_patience}"
                )
                log.info(msg_pat)
                print(msg_pat)

            if self.progress_plot_frequency and self.progress_plot_frequency > 0 and (epoch + 1) % self.progress_plot_frequency == 0:
                utils.plotLoss(
                    pGeneratorLossValueLists=[self.__diff_train_loss_epochs, self.__diff_val_loss_epochs],
                    pDiscLossValueLists=[],
                    pGeneratorLossNameList=["training", "validation"],
                    pDiscLossNameList=[],
                    pFilename=self.progress_plot_name,
                    useLogscaleList=[True, False],
                    pYLabel="diffusion loss",
                    pLegendTitle="Denoiser",
                )
                _save_loss_files(epoch)
                self.denoiser.save(filepath=os.path.join(self.log_dir, "denoiser_{:05d}.keras".format(epoch)), save_format="keras")
                self.ckpt_epoch.assign(epoch + 1)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            if early_stopping_patience is not None and early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                stop_msg = (
                    f"[EARLY-STOPPING] Stopping early at epoch {epoch + 1}/{epochs} "
                    f"(no improvement in val_total_loss for {patience_counter} epochs; best={best_val_loss:.6f} at epoch {best_epoch})."
                )
                log.info(stop_msg)
                print(stop_msg)
                break

        if int(self.ckpt_epoch.numpy()) != (last_epoch + 1):
            self.ckpt_epoch.assign(last_epoch + 1)
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        if best_weights is not None:
            self.denoiser.set_weights(best_weights)
            restore_msg = f"[MODEL-RESTORE] Restored best denoiser weights from epoch {best_epoch} (val_total_loss={best_val_loss:.6f})."
            log.info(restore_msg)
            print(restore_msg)

        utils.plotLoss(
            pGeneratorLossValueLists=[self.__diff_train_loss_epochs, self.__diff_val_loss_epochs],
            pDiscLossValueLists=[],
            pGeneratorLossNameList=["training", "validation"],
            pDiscLossNameList=[],
            pFilename=self.progress_plot_name,
            useLogscaleList=[True, False],
            pYLabel="diffusion loss",
            pLegendTitle="Denoiser",
        )
        _save_loss_files(last_epoch)
        self.denoiser.save(filepath=os.path.join(self.log_dir, "denoiser_{:05d}.keras".format(last_epoch)), save_format="keras")
        self.denoiser.save(filepath=os.path.join(self.log_dir, "denoiser_final.keras"), save_format="keras")

    def plotModels(self, pOutputPath: str, pFigureFileFormat: str):
        denoiserPlotName = os.path.join(pOutputPath, "denoiserModel.{:s}".format(pFigureFileFormat))
        unetCorePlotName = os.path.join(pOutputPath, "unetCoreModel.{:s}".format(pFigureFileFormat))
        condEmbPlotName = os.path.join(pOutputPath, "conditionEncoderModel.{:s}".format(pFigureFileFormat))
        modelsToPlot = [
            (self.denoiser, denoiserPlotName),
            (self.unet_denoiser, unetCorePlotName),
        ]
        if self.use_embedding and self.condition_encoder is not None:
            modelsToPlot.append((self.condition_encoder, condEmbPlotName))
        for model, name in modelsToPlot:
            try:
                tf.keras.utils.plot_model(model, show_shapes=True, to_file=name)
            except (ImportError, OSError) as ex:
                msg = ("skipping the architecture diagram {:s}: {:s}. "
                       "Install pydot and graphviz if you want it; training is "
                       "unaffected.").format(name, str(ex).split("\n")[0])
                print(msg)
                break

    def predict(self, test_ds, steps_per_record):
        predictedArray = []
        if self.denoiser is None:
            raise ValueError("Denoiser is None. Please call load_denoiser() first.")
        print(f"DEBUG: Starting DDIM prediction ({self.sample_steps} steps). Steps arg: {steps_per_record}.")
        for i, batch in tqdm(enumerate(test_ds), desc="Predicting (DDIM)"):
            input_data = HiCGAN._inputFromBatch(batch)
            predBatch = self.predictionStep(input_batch=input_data, training=False)
            val = predBatch.numpy()
            if i == 0:
                print(f"DEBUG: First batch raw output shape: {val.shape}")
            if val.ndim == 4 and val.shape[-1] == 1:
                val = np.squeeze(val, axis=-1)
            predictedArray.append(val)
            del predBatch, val
        if len(predictedArray) == 0:
            print("WARNING: predictedArray is empty!")
            return np.array([])
        total_prediction = np.concatenate(predictedArray, axis=0)
        print(f"DEBUG: Final Reference Shape returned: {total_prediction.shape}")
        return total_prediction

    def predict_stream(self, test_ds):
        '''Memory-bounded streaming DDIM prediction.'''
        if self.denoiser is None:
            raise ValueError("Denoiser is None. Please call load_denoiser() first.")
        for batch in test_ds:
            input_data = HiCGAN._inputFromBatch(batch)
            val = self.predictionStep(input_batch=input_data, training=False).numpy()
            if val.ndim == 4 and val.shape[-1] == 1:
                val = np.squeeze(val, axis=-1)
            yield val

    def predict_stream_triu(self, test_ds, window_size):
        '''Stream predictions already reduced to the upper triangle, float16.'''
        if self.denoiser is None:
            raise ValueError("Denoiser is None. Please call load_denoiser() first.")
        r, c = np.triu_indices(window_size)
        flat = tf.constant(r * window_size + c, dtype=tf.int32)

        @tf.function
        def _step(inp):
            out = self.predictionStep(input_batch=inp, training=False)
            if out.shape.rank == 4 and out.shape[-1] == 1:
                out = tf.squeeze(out, axis=-1)
            out = tf.reshape(out, [tf.shape(out)[0], window_size * window_size])
            return tf.cast(tf.gather(out, flat, axis=1), tf.float16)

        for batch in test_ds:
            input_data = HiCGAN._inputFromBatch(batch)
            yield _step(input_data).numpy()

    @staticmethod
    def _inputFromBatch(batch):
        if isinstance(batch, (tuple, list)):
            if isinstance(batch[0], dict) and "factorData" in batch[0]:
                return batch[0]["factorData"]
            return batch[0]
        if isinstance(batch, dict):
            return batch.get("factorData", batch)
        return batch

    def predict_band(self, test_ds, window_size, matrix_size, flanking_size,
                     window_starts=None):
        '''Predict a chromosome via DDIM and sum the overlapping windows on the GPU.'''
        if self.denoiser is None:
            raise ValueError("Denoiser is None. Please call load_denoiser() first.")
        W = int(window_size)
        matrixSize = int(matrix_size)
        flanking = int(flanking_size)
        starts = None if window_starts is None else np.asarray(window_starts, dtype=np.int32)
        r, c = np.triu_indices(W)
        flatWindow = tf.constant((r * W + c).astype(np.int32))
        flatBase = tf.constant(((c - r).astype(np.int64) * matrixSize + r).astype(np.int32))

        band = tf.Variable(tf.zeros([W * matrixSize], dtype=tf.float32), trainable=False)

        @tf.function
        def _step(inp, offsets):
            out = self.predictionStep(input_batch=inp, training=False)
            if out.shape.rank == 4 and out.shape[-1] == 1:
                out = tf.squeeze(out, axis=-1)
            out = tf.reshape(out, [tf.shape(out)[0], W * W])
            tri = tf.cast(tf.gather(out, flatWindow, axis=1), tf.float32)
            idx = flatBase[tf.newaxis, :] + offsets[:, tf.newaxis]
            band.scatter_nd_add(tf.reshape(idx, [-1, 1]), tf.reshape(tri, [-1]))

        pos = 0
        usedStarts = []
        for batch in test_ds:
            input_data = HiCGAN._inputFromBatch(batch)
            n = int(input_data.shape[0])
            if starts is None:
                offsets = np.arange(pos, pos + n, dtype=np.int32) + flanking
            else:
                if pos + n > starts.shape[0]:
                    raise ValueError(
                        "more samples in the dataset ({:d}+) than window starts ({:d})".format(
                            pos + n, starts.shape[0]))
                offsets = starts[pos:pos + n] + flanking
            if int(offsets.max()) + W > matrixSize:
                raise ValueError(
                    "window at {:d} exceeds the {:d} bins of the chromosome".format(
                        int(offsets.max()), matrixSize))
            usedStarts.append(offsets)
            _step(input_data, tf.convert_to_tensor(offsets))
            pos += n
        if starts is not None and pos != starts.shape[0]:
            print("Warning: predicted {:d} windows but {:d} starts were given".format(
                pos, starts.shape[0]), flush=True)
        result = band.numpy().reshape(W, matrixSize)
        del band
        allStarts = (np.concatenate(usedStarts) if usedStarts
                     else np.zeros(0, dtype=np.int32))
        return result, allStarts

    def load_denoiser(self, trainedModelPath: str):
        '''Load a trained diffusion denoiser model (.keras or checkpoint dir) and rebind cached sub-models.'''
        try:
            print(f"Attempting to load denoiser model from: {trainedModelPath}")
            resolved_path = trainedModelPath
            if resolved_path and os.path.isdir(resolved_path):
                for cand_name in ("denoiser_best.keras", "denoiser_final.keras"):
                    cand_p = os.path.join(resolved_path, cand_name)
                    if os.path.exists(cand_p):
                        resolved_path = cand_p
                        break

            custom_layers = {
                "CustomReshapeLayer": CustomReshapeLayer,
                "SplitChromatinStreamsLayer": SplitChromatinStreamsLayer,
                "Pairwise2DEmbeddingLayer": Pairwise2DEmbeddingLayer,
                "Symmetrize2DLayer": Symmetrize2DLayer,
                "SpatialTimeEncoding": SpatialTimeEncoding,
                "GroupNormalization": tf.keras.layers.GroupNormalization,
            }

            if os.path.isdir(resolved_path) or "ckpt" in os.path.basename(resolved_path):
                ckpt = tf.train.Checkpoint(denoiser=self.denoiser)
                latest = tf.train.latest_checkpoint(resolved_path) if os.path.isdir(resolved_path) else resolved_path
                if latest:
                    print(f"Restoring denoiser from checkpoint: {latest}")
                    ckpt.restore(latest).expect_partial()
                else:
                    raise FileNotFoundError(f"No denoiser model or checkpoint found at {resolved_path}")
            else:
                self.denoiser = tf.keras.models.load_model(
                    filepath=resolved_path,
                    custom_objects=custom_layers,
                    safe_mode=False,
                )

            # Rebind condition_encoder & unet_denoiser sub-models so predictionStep caches condition_encoder
            try:
                self.unet_denoiser = self.denoiser.get_layer("UNet_Diffusion_Core")
            except ValueError:
                pass
            if self.use_embedding:
                try:
                    self.condition_encoder = self.denoiser.get_layer("CNN-embedding")
                except ValueError:
                    pass
            try:
                ste_layer = self.unet_denoiser.get_layer("spatial_time_encoding")
                loaded_t = max(1, int(round(float(ste_layer.timesteps))))
                if loaded_t != self.timesteps:
                    log.info("[MODEL-LOAD] Syncing diffusion timesteps from %d -> %d based on saved SpatialTimeEncoding.", self.timesteps, loaded_t)
                    self.timesteps = loaded_t
                    betas_steps = tf.cast(tf.linspace(1e-4, 0.02, self.timesteps), tf.float32)
                    self.betas = tf.concat([[0.0], betas_steps], axis=0)
                    self.alphas = 1.0 - self.betas
                    self.alphas_bar = tf.math.cumprod(self.alphas)
            except Exception:
                pass

            print("Denoiser model loaded successfully. Running diagnostics...")
            all_weights = self.denoiser.get_weights()
            if len(all_weights) > 0:
                total_weight_sum = np.sum([np.sum(np.abs(w)) for w in all_weights])
                has_nan = np.any([np.isnan(np.sum(w)) for w in all_weights])
                print(f" -> Total sum of weights: {total_weight_sum}")
                if has_nan:
                    raise ValueError("CRITICAL: The loaded model contains NaN weights.")
                if total_weight_sum == 0:
                    raise ValueError("CRITICAL: All model weights are ZERO.")
            print(" -> Diagnostics passed. Weights look valid.")
        except Exception as e:
            msg = f"Failed to load denoiser.\nDetail: {str(e)}"
            raise ValueError(msg)

    # Alias for backward compatibility
    loaddenoiser = load_denoiser
    loadGenerator = load_denoiser


@tf.keras.utils.register_keras_serializable(package="hicgan")
class Symmetrize2DLayer(tf.keras.layers.Layer):
    """Symmetrize a 4D (batch, H, W, C) tensor along its spatial axes: 0.5 * (X + X^T)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x_T = tf.transpose(inputs, perm=[0, 2, 1, 3])
        return 0.5 * (inputs + x_T)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="hicgan")
class SplitChromatinStreamsLayer(tf.keras.layers.Layer):
    """
    Split 1D (+ optional packed 2D pair_features) factorData of shape (batch, 3*W, number_factors) into:
      - f_row:       (batch, 3*W, n_chrom)
      - f_col:       (batch, 3*W, n_chrom)
      - pos_row:     (batch, 3*W, 1)
      - pos_col:     (batch, 3*W, 1)
      - pair_mat_2d: (batch, W, W, 1)
    """
    def __init__(self, window_size: int, number_factors: int, is_grid_mode: bool = False, has_pair_2d: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.number_factors = int(number_factors)
        self.is_grid_mode = bool(is_grid_mode)
        self.has_pair_2d = bool(has_pair_2d)

    def call(self, inputs):
        x = inputs
        if x.shape.rank == 4:
            x = tf.squeeze(x, axis=-1)
        W = self.window_size
        if self.has_pair_2d and self.number_factors > W:
            c_1d = self.number_factors - W
            x_1d = x[:, :, :c_1d]
            pair_mat_2d = tf.expand_dims(x[:, W : 2 * W, c_1d:], axis=-1)
        else:
            c_1d = self.number_factors
            x_1d = x
            z_base = x[:, W : 2 * W, :1] * 0.0
            pair_mat_2d = tf.expand_dims(z_base, axis=2) + tf.expand_dims(z_base, axis=1)

        if self.is_grid_mode and c_1d >= 4 and (c_1d % 2 == 0):
            half_ch = c_1d // 2
            n_chrom = half_ch - 1
            f_row = x_1d[:, :, :n_chrom]
            pos_row = x_1d[:, :, n_chrom:half_ch]
            f_col = x_1d[:, :, half_ch : half_ch + n_chrom]
            pos_col = x_1d[:, :, half_ch + n_chrom : 2 * half_ch]
        else:
            f_row = x_1d
            f_col = x_1d
            pos_row = tf.zeros_like(x_1d[:, :, :1])
            pos_col = tf.zeros_like(x_1d[:, :, :1])
        return f_row, f_col, pos_row, pos_col, pair_mat_2d

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "window_size": self.window_size,
            "number_factors": self.number_factors,
            "is_grid_mode": self.is_grid_mode,
            "has_pair_2d": self.has_pair_2d,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="hicgan")
class Pairwise2DEmbeddingLayer(tf.keras.layers.Layer):
    """
    Construct a 2D (batch, W, W, C_pair) pairwise chromatin + genomic distance (+ optional 2D pair_features) representation.
    """
    def __init__(self, window_size: int, is_grid_mode: bool = False, has_pair_2d: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.is_grid_mode = bool(is_grid_mode)
        self.has_pair_2d = bool(has_pair_2d)
        idx = np.arange(self.window_size, dtype=np.float32)
        diag_dist_bins = np.abs(idx[:, None] - idx[None, :])[None, :, :, None]
        self._diag_dist_bins = tf.constant(diag_dist_bins, dtype=tf.float32)

    def call(self, inputs):
        f_row, f_col, h_row_full, h_col_full, h_row_stride, h_col_stride, pos_row, pos_col, pair_mat_2d = inputs
        W = self.window_size
        compute_dtype = h_row_full.dtype

        f_row_c = tf.cast(f_row[:, W : 2 * W, :], compute_dtype)
        f_col_c = tf.cast(f_col[:, W : 2 * W, :], compute_dtype)
        h_row_c = h_row_full[:, W : 2 * W, :]
        h_col_c = h_col_full[:, W : 2 * W, :]

        e_row = tf.concat([h_row_c, h_row_stride, f_row_c], axis=-1)
        e_col = tf.concat([h_col_c, h_col_stride, f_col_c], axis=-1)

        E_row = tf.tile(tf.expand_dims(e_row, axis=2), [1, 1, W, 1])
        E_col = tf.tile(tf.expand_dims(e_col, axis=1), [1, W, 1, 1])

        if self.is_grid_mode:
            p_row = tf.cast(pos_row[:, W : 2 * W, 0], tf.float32)
            p_col = tf.cast(pos_col[:, W : 2 * W, 0], tf.float32)
            dist_norm = tf.abs(tf.expand_dims(p_col, axis=1) - tf.expand_dims(p_row, axis=2))
            dist_norm = tf.expand_dims(dist_norm, axis=-1)
            dist_bins = dist_norm * float(W)
        else:
            batch_sz = tf.shape(f_row)[0]
            dist_bins = tf.tile(self._diag_dist_bins, [batch_sz, 1, 1, 1])
            dist_norm = dist_bins / float(W)

        d_inv1 = 1.0 / (1.0 + dist_bins / 2.0)
        d_inv2 = 1.0 / (1.0 + dist_bins / 8.0)
        d_exp = tf.exp(-dist_norm)
        d_diag = tf.exp(-tf.square(dist_bins / 1.5))

        dist_only = tf.cast(tf.concat([d_inv1, d_inv2, d_exp, d_diag], axis=-1), compute_dtype)
        d_gate = tf.cast(d_inv2, compute_dtype)

        E_prod = E_row * E_col
        E_prod_gated = E_prod * d_gate
        E_sum = 0.5 * (E_row + E_col)
        E_diff = tf.abs(E_row - E_col)

        if self.has_pair_2d:
            pair_2d_feat = tf.cast(pair_mat_2d, compute_dtype)
            E_prod_kmer = E_prod * pair_2d_feat
            dist_2d = tf.concat([dist_only, pair_2d_feat], axis=-1)
            pair_2d = tf.concat([E_prod, E_prod_gated, E_prod_kmer, E_sum, E_diff, E_row, E_col, dist_2d], axis=-1)
        else:
            dist_2d = dist_only
            pair_2d = tf.concat([E_prod, E_prod_gated, E_sum, E_diff, E_row, E_col, dist_2d], axis=-1)
        return pair_2d, dist_2d

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "window_size": self.window_size,
            "is_grid_mode": self.is_grid_mode,
            "has_pair_2d": self.has_pair_2d,
        })
        return cfg


class CustomReshapeLayer(tf.keras.layers.Layer):
    '''
    reshape a 1D tensor such that it represents
    the upper triangular part of a square 2D matrix with shape (matsize, matsize)
    '''
    def __init__(self, matsize, **kwargs):
        super(CustomReshapeLayer, self).__init__(**kwargs)
        self.matsize = matsize
        self.triu_indices = [[x, y] for x, y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1])]

    def call(self, inputs):
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)

    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(
            self.triu_indices,
            values=inputVec,
            dense_shape=[self.matsize, self.matsize],
        )
        return tf.sparse.to_dense(sparseTriuTens)

    def get_config(self):
        return {"matsize": self.matsize}
