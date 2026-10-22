import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU, Flatten, Dense
from keras.layers import GroupNormalization
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from tqdm import tqdm
from . import utils
import re

#implementation of adapted pix2pix cGAN
#modified from tensorflow tutorial https://www.tensorflow.org/tutorials/generative/pix
#also see: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

@keras.saving.register_keras_serializable(package="HiCGAN")
class SpatialTimeEncoding(keras.layers.Layer):
    def __init__(self, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = float(timesteps)

    def call(self, inputs):
        dna, t = inputs
        t_scaled = t / self.timesteps
        t_expanded = tf.reshape(t_scaled, [-1, 1, 1, 1])
        return tf.broadcast_to(t_expanded, tf.shape(dna))

    def get_config(self):
        config = super().get_config()
        config.update({"timesteps": self.timesteps})
        return config


class HiCGAN():
    def __init__(self, log_dir: str,
                     number_factors: int,
                     timesteps: int = 1000,  # Diffusion step count
                     loss_weight_pixel: float = 1.0,
                     input_size: int = 256,
                     plot_frequency: int = 20,
                     plot_type: str = "png",
                     learning_rate: float = 2e-5,
                     adam_beta_1: float = 0.5,
                     pretrained_model_path: str = "",
                     scope=None,
                     mixed_precision: bool = False,
                     restore_checkpoint: bool = False):
            super().__init__()

            self.mixed_precision = mixed_precision
            self.OUTPUT_CHANNELS = 1
            self.INPUT_CHANNELS = 1
            self.input_size = input_size
            self.number_factors = number_factors
            self.loss_weight_pixel = loss_weight_pixel

            # NOISE SCHEDULER
            self.timesteps = timesteps
            beta_start = 1e-4
            beta_end = 0.02

            # Set beta_0 = 0.0 (alpha_bar_0 = 1.0) and linearly schedule betas for t = 1..timesteps
            betas_steps = tf.cast(tf.linspace(beta_start, beta_end, self.timesteps), tf.float32)
            self.betas = tf.concat([[0.0], betas_steps], axis=0)

            self.alphas = 1.0 - self.betas

            self.alphas_bar = tf.math.cumprod(self.alphas)

            self.denoiser_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=adam_beta_1, name="Adam_UNet")

            if self.mixed_precision:
                self.denoiser_optimizer = keras.mixed_precision.LossScaleOptimizer(self.denoiser_optimizer)

            #Model Start
            self.denoiser = self.denoiser()

            self.log_dir = log_dir
            self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

            self.checkpoint = tf.train.Checkpoint(denoiser_optimizer=self.denoiser_optimizer,
                                                  denoiser=self.denoiser)
            self.start_epoch = 0
            if restore_checkpoint:
                latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if latest_checkpoint:
                    print(f"Restoring from checkpoint: {latest_checkpoint}")
                    self.checkpoint.restore(latest_checkpoint)
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
                else:
                    print("No checkpoint found. Starting from scratch.")

            self.plot_type = plot_type if plot_type in ["png", "pdf", "svg"] else "png"
            self.progress_plot_name = os.path.join(self.log_dir, "lossOverEpochs.{:s}".format(self.plot_type))
            self.progress_plot_frequency = plot_frequency
            self.example_plot_frequency = 5

            self.__gen_train_loss_epochs = []
            self.__gen_val_loss_epochs = []
            self.__gen_train_loss_batches = []
            self.__gen_val_loss_batches = []

            self.__epoch_counter = 0
            self.__batch_counter = 0
            if scope is None:
                scope = tf.distribute.get_strategy()
            self.scope = scope

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            # result.add(BatchNormalization())
            result.add(GroupNormalization(groups=filters, axis=-1))
        result.add(LeakyReLU(negative_slope=0.2))
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        # result.add(BatchNormalization())
        result.add(GroupNormalization(groups=filters, axis=-1))
        if apply_dropout:
            result.add(Dropout(0.5))
        result.add(ReLU())
        return result

    def denoiser(self):
        # 3 different inputs
        noisy_dna = keras.layers.Input(shape=[self.input_size, self.input_size, 1], name="noisy_dna")
        rna_condition = keras.layers.Input(shape=[self.input_size, self.input_size, 1], name="rna_condition")
        time_step = keras.layers.Input(shape=[1],
                                          name="time_step")

        time_channel = SpatialTimeEncoding(timesteps=self.timesteps)([noisy_dna, time_step])

        # Concat 3 channels as 1 block
        x = keras.layers.Concatenate(axis=-1)([noisy_dna, rna_condition, time_channel])

        # Dynamic Architecture
        num_downsamples = int(np.round(np.log2(self.input_size)))
        down_filters = [min(64 * (2 ** i), 512) for i in range(num_downsamples)]
        up_filters = [min(64 * (2 ** i), 512) for i in reversed(range(num_downsamples - 1))]

        down_stack = []
        for i, f in enumerate(down_filters):
            apply_norm = (i > 0 and i < num_downsamples - 1)
            down_stack.append(HiCGAN.downsample(f, 4, apply_batchnorm=apply_norm))

        up_stack = []
        for i, f in enumerate(up_filters):
            #prevent overfitting
            apply_drop = (i < 3)
            up_stack.append(HiCGAN.upsample(f, 4, apply_dropout=apply_drop))

        initializer = tf.random_normal_initializer(0., 0.02)

        #Downsampling
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = list(reversed(skips[:-1]))

        #Upsampling
        for up, skip in zip(up_stack, skips):
            x = up(x)
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                x = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(x)
            x = keras.layers.Concatenate()([x, skip])

        # Output: unconstrained pixel-wise noise prediction (symmetry is enforced on final denoised x_0)
        last = keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                               kernel_initializer=initializer)
        x = last(x)
        if x.shape[1] != self.input_size or x.shape[2] != self.input_size:
            x = tf.keras.layers.Resizing(self.input_size, self.input_size)(x)

        return keras.Model(inputs=[noisy_dna, rna_condition, time_step], outputs=x, name="UNet_Diffusion")


    @tf.function
    def distributed_train_step(self, input_data):
        rna_matrix = input_data[0]["factorData"]
        real_dna_matrix = input_data[1]["out_matrixData"]

        per_replica_losses = self.scope.run(self.train_step, args=(rna_matrix, real_dna_matrix,))
        # Average the loss across replicas so multi-GPU training reports mean MSE
        return self.scope.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    @tf.function
    def train_step(self, rna_matrix, real_dna_matrix):
        batch_size = tf.shape(real_dna_matrix)[0]

        t = tf.random.uniform(shape=[batch_size, 1], minval=1, maxval=self.timesteps + 1, dtype=tf.int32)

        noise = tf.random.normal(shape=tf.shape(real_dna_matrix))

        # FORWARD PROCESS
        alpha_bar_t = tf.gather(self.alphas_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

        # DDPM
        noisy_dna = tf.sqrt(alpha_bar_t) * real_dna_matrix + tf.sqrt(1.0 - alpha_bar_t) * noise

        with tf.GradientTape() as tape:
            predicted_noise = self.denoiser([noisy_dna, rna_matrix, tf.cast(t, tf.float32)], training=True)

            loss = tf.reduce_mean(tf.square(noise - predicted_noise))  #mse

            if self.mixed_precision:
                scaled_loss = self.denoiser_optimizer.get_scaled_loss(loss)

        #BACKPROPAGATION
        if self.mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.denoiser.trainable_variables)
            gradients = self.denoiser_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.denoiser.trainable_variables)

        self.denoiser_optimizer.apply_gradients(zip(gradients, self.denoiser.trainable_variables))

        return loss

    @tf.function
    def validationStep(self, rna_matrix, real_dna_matrix):
        batch_size = tf.shape(real_dna_matrix)[0]

        t = tf.random.uniform(shape=[batch_size, 1], minval=1, maxval=self.timesteps + 1, dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(real_dna_matrix))

        alpha_bar_t = tf.gather(self.alphas_bar, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])
        noisy_dna = tf.sqrt(alpha_bar_t) * real_dna_matrix + tf.sqrt(1.0 - alpha_bar_t) * noise

        #prediction
        predicted_noise = self.denoiser([noisy_dna, rna_matrix, tf.cast(t, tf.float32)], training=False)
        loss = tf.reduce_mean(tf.square(noise - predicted_noise))
        return loss

    def generate_images(self, model, test_input, tar, epoch: int):
        rna_condition = test_input["factorData"][:1]
        real_dna = tar["out_matrixData"][:1]
        batch_size = 1

        x_t = tf.random.normal(shape=(batch_size, self.input_size, self.input_size, 1))

        for t_step in reversed(range(1, self.timesteps + 1)):
            t_batch = tf.fill([batch_size, 1], tf.cast(t_step, tf.float32))
            predicted_noise = model([x_t, rna_condition, t_batch], training=False)

            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alphas_bar[t_step]
            alpha_bar_prev = self.alphas_bar[t_step - 1]
            beta_t = self.betas[t_step]

            # Estimate x_0 and clip to [0, 1] RNA target range for reverse-diffusion stability
            x0_pred = (x_t - tf.sqrt(1.0 - alpha_bar_t) * predicted_noise) / tf.sqrt(alpha_bar_t)
            x0_pred = tf.clip_by_value(x0_pred, 0.0, 1.0)

            posterior_mean = (
                (tf.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)) * x0_pred +
                (tf.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * x_t
            )
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

            if t_step > 1:
                noise = tf.random.normal(shape=tf.shape(x_t))
                x_t = posterior_mean + tf.sqrt(posterior_var) * noise
            else:
                x_t = posterior_mean

        # Enforce symmetry and valid [0, 1] range on the final denoised matrix x_0
        prediction = 0.5 * (x_t + tf.transpose(x_t, perm=[0, 2, 1, 3]))
        prediction = tf.clip_by_value(prediction, 0.0, 1.0)
        pred_mse = tf.reduce_mean(tf.square(real_dna[0] - prediction[0]))

        figname = "testpred_epoch_{:05d}.png".format(epoch)
        figname = os.path.join(self.log_dir, figname)
        # Plot Random Input condition, Ground Truth RNA matrix, and Predicted RNA matrix
        display_list = [rna_condition[0], real_dna[0], prediction[0]]
        titleList = ['Random Input', 'Ground Truth RNA', 'Predicted RNA (MSE: {:.5f})'.format(pred_mse)]

        fig1, axs1 = plt.subplots(1, len(display_list), figsize=(15, 5))
        for i in range(len(display_list)):
            img = np.squeeze(display_list[i].numpy() if hasattr(display_list[i], 'numpy') else display_list[i])
            im = axs1[i].imshow(img, cmap='coolwarm')
            axs1[i].set_title(titleList[i])
            fig1.colorbar(im, ax=axs1[i], fraction=0.046, pad=0.04)
        fig1.tight_layout()
        fig1.savefig(figname)
        plt.close(fig1)
        del fig1, axs1

    def fit(self, train_ds, epochs, test_ds, steps_per_epoch: int,
            validation_steps: int = 0):
        """validation_steps > 0 evaluates only that many validation batches.

        Validation runs the WHOLE validation set every epoch and, unlike the
        training step, is not distributed across replicas -- so its cost is
        fixed while the training half gets faster with every GPU added. On the
        chr21 pilot it was ~145 s against 94 s of actual training on two GPUs;
        on eight it would dominate the epoch completely. A few hundred windows
        track the loss curve just as well.
        """
        distributed_dataset = self.scope.experimental_distribute_dataset(train_ds)
        
        if self.start_epoch > 0 and self.start_epoch < epochs:
            print(f"Resuming training from epoch {self.start_epoch}.")
        elif self.start_epoch >= epochs:
            print(f"Checkpoint epoch {self.start_epoch} is greater than or equal to total epochs {epochs}. Starting from scratch.")
            exit(1)

        vis_sample_input = None
        vis_sample_target = None
        # Select a validation sample with non-zero Ground Truth RNA interactions for visualization
        for test_in, test_tar in test_ds:
            targets = test_tar["out_matrixData"]
            sums = tf.reduce_sum(tf.abs(targets), axis=[1, 2, 3])
            max_idx = int(tf.argmax(sums))
            if float(sums[max_idx]) > 0.0:
                vis_sample_input = {k: v[max_idx:max_idx+1] for k, v in test_in.items()}
                vis_sample_target = {k: v[max_idx:max_idx+1] for k, v in test_tar.items()}
                break

        if vis_sample_input is None:
            for test_in, test_tar in test_ds.take(1):
                vis_sample_input = {k: v[:1] for k, v in test_in.items()}
                vis_sample_target = {k: v[:1] for k, v in test_tar.items()}

        for epoch in range(self.start_epoch, epochs):
            #generate sample output
            if epoch % self.example_plot_frequency == 0 and vis_sample_input is not None:
                self.generate_images(self.denoiser, vis_sample_input, vis_sample_target, epoch)
            
            
            train_pbar = tqdm(total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            train_samples_in_epoch = 0

            for distributed_input in distributed_dataset:
                train_samples_in_epoch += 1

                gen_loss = self.distributed_train_step(distributed_input)
                self.__gen_train_loss_batches.append(gen_loss)

                train_bar_postfixDict = {"Loss (MSE)": "{:.4f}".format(self.__gen_train_loss_batches[-1])}
                if len(self.__gen_val_loss_epochs) > 0:
                    train_bar_postfixDict["Val_Loss"] = "{:.3f}".format(self.__gen_val_loss_epochs[-1])
                train_pbar.set_postfix(train_bar_postfixDict)
                train_pbar.update(1)
                self.__batch_counter += 1

            train_pbar.close()
            self.__gen_train_loss_epochs.append(np.mean(self.__gen_train_loss_batches[-train_samples_in_epoch:]))

            # Validation
            validation_samples_in_epoch = 0
            vds = test_ds.take(validation_steps) if validation_steps else test_ds
            for input_image, target in vds:
                val_loss = self.validationStep(input_image["factorData"], target["out_matrixData"])
                self.__gen_val_loss_batches.append(val_loss)
                validation_samples_in_epoch += 1
            if validation_samples_in_epoch > 0:
                self.__gen_val_loss_epochs.append(np.mean(self.__gen_val_loss_batches[-validation_samples_in_epoch:]))

            #count the epoch
            self.__epoch_counter += 1

            # saving the model periodically
            if (epoch + 1) % self.progress_plot_frequency == 0:
                #plot loss
                utils.plotLoss(pdenoiserLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                              pDiscLossValueLists=[],
                              pdenoiserLossNameList=["training", "validation"],
                              pDiscLossNameList=[],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
                np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=self.__gen_train_loss_batches, 
                                    genLossVal=self.__gen_val_loss_batches)
                self.denoiser.save(filepath=os.path.join(self.log_dir, "denoiser_{:05d}.keras".format(epoch)), save_format="keras")

            self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        # Final save after training completes
        utils.plotLoss(pdenoiserLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                      pDiscLossValueLists=[],
                      pdenoiserLossNameList=["training", "validation"],
                      pDiscLossNameList=[],
                      pFilename=self.progress_plot_name,
                      useLogscaleList=[True, False])
        np.savez(os.path.join(self.log_dir, "lossValues_final.npz"), 
                            genLossTrain=self.__gen_train_loss_batches, 
                            genLossVal=self.__gen_val_loss_batches)
        self.denoiser.save(filepath=os.path.join(self.log_dir, "denoiser_final.keras"), save_format="keras")

    def plotModels(self, pOutputPath: str, pFigureFileFormat: str):
        denoiserPlotName = "denoiserModel.{:s}".format(pFigureFileFormat)
        denoiserPlotName = os.path.join(pOutputPath, denoiserPlotName)
        try:
            tf.keras.utils.plot_model(self.denoiser, show_shapes=True, to_file=denoiserPlotName)
        except (ImportError, OSError) as ex:
            msg = ("skipping the architecture diagram {:s}: {:s}. "
                   "Install pydot and graphviz if you want it; training is "
                   "unaffected.").format(denoiserPlotName, str(ex).split("\n")[0])
            print(msg)

    # def predict(self, test_ds, steps_per_record):
    #     predictedArray = []
    #     for batch in tqdm(test_ds, desc="Predicting", total=steps_per_record):
    #         predBatch = self.predictionStep(input_batch=batch, training=False)
    #         # Extract and append only the needed slice to reduce memory copies
    #         predictedArray.append(predBatch[:, :, :, 0].numpy())
    #         # Free memory from the prediction batch
    #         del predBatch
    #     # Concatenate along batch dimension instead of converting list
    #     predictedArray = np.concatenate(predictedArray, axis=0)
    #     return predictedArray

    def predict(self, test_ds, steps_per_record):
        predictedArray = []

        if self.denoiser is None:
            raise ValueError("denoiser is None. Please call loaddenoiser() first.")

        # Debug print to verify request
        print(f"DEBUG: Starting prediction. Steps arg: {steps_per_record}.")

        # Iterate over dataset
        # We don't use 'total=steps_per_record' to avoid confusing tqdm if steps is wrong
        for i, batch in tqdm(enumerate(test_ds), desc="Predicting"):
            
            # --- 1. Robust Input Extraction ---
            input_data = None
            
            # Check for Tuple/List (common in tf.data: (input, target))
            if isinstance(batch, (tuple, list)):
                # If the first element is a dict (feature dict), use that
                if isinstance(batch[0], dict) and "factorData" in batch[0]:
                    input_data = batch[0]["factorData"]
                else:
                    input_data = batch[0] # Assume index 0 is input
            # Check for Dict (direct feature dict)
            elif isinstance(batch, dict):
                input_data = batch.get("factorData", batch)
            # Fallback (tensor)
            else:
                input_data = batch

            # --- 2. Prediction ---
            # training=False is critical
            predBatch = self.predictionStep(input_batch=input_data, training=False)
            
            # --- 3. Cleaning & Formatting ---
            val = predBatch.numpy()
            
            # Debug first batch shape to ensure model is outputting what we expect
            if i == 0:
                print(f"DEBUG: First batch raw output shape: {val.shape}")

            # Squeeze channel dimension if present: (Batch, 64, 64, 1) -> (Batch, 64, 64)
            if val.ndim == 4 and val.shape[-1] == 1:
                val = np.squeeze(val, axis=-1)
            
            predictedArray.append(val)

            # Cleanup
            del predBatch
            del val

        # --- 4. Aggregation ---
        if len(predictedArray) == 0:
            print("WARNING: predictedArray is empty!")
            return np.array([])

        # Concatenate all batches into one large array: (Total_Samples, 64, 64)
        total_prediction = np.concatenate(predictedArray, axis=0)
        
        print(f"DEBUG: Final Reference Shape returned: {total_prediction.shape}")
        
        return total_prediction
    
    def predict_stream(self, test_ds):
        '''
        Memory-bounded streaming prediction.

        Iterates the dataset exactly once and yields the (squeezed) prediction
        for one batch at a time as a numpy array of shape (batch, W, W). The
        caller is expected to reduce each batch immediately (e.g. extract the
        upper triangle as float16), so the full-chromosome float32 stack is
        never materialised. This avoids both the giant concatenation in
        predict() and the quadratic dataset.skip() re-decoding used by the old
        save-memory prediction loop.
        '''
        if self.denoiser is None:
            raise ValueError("denoiser is None. Please call loaddenoiser() first.")
        for batch in test_ds:
            # robust input extraction (mirrors predict())
            if isinstance(batch, (tuple, list)):
                if isinstance(batch[0], dict) and "factorData" in batch[0]:
                    input_data = batch[0]["factorData"]
                else:
                    input_data = batch[0]
            elif isinstance(batch, dict):
                input_data = batch.get("factorData", batch)
            else:
                input_data = batch
            val = self.predictionStep(input_batch=input_data, training=False).numpy()
            if val.ndim == 4 and val.shape[-1] == 1:
                val = np.squeeze(val, axis=-1)
            yield val

    def predict_stream_triu(self, test_ds, window_size):
        '''Stream predictions already reduced to the upper triangle, float16.

        predict_stream() copies the whole (batch, W, W) float32 tensor to the
        host and leaves the caller to gather the upper triangle there. Both the
        transfer and the gather happen between two GPU calls, so the card waits
        through them: measured utilisation was 49% with a quarter of samples
        below 5%.

        Doing the gather inside the graph halves the bytes transferred, since
        only the triangle crosses the bus and it crosses as float16 rather than
        float32, and removes the host-side gather entirely.
        '''
        if self.denoiser is None:
            raise ValueError("denoiser is None. Please call loaddenoiser() first.")
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
            if isinstance(batch, (tuple, list)):
                if isinstance(batch[0], dict) and "factorData" in batch[0]:
                    input_data = batch[0]["factorData"]
                else:
                    input_data = batch[0]
            elif isinstance(batch, dict):
                input_data = batch.get("factorData", batch)
            else:
                input_data = batch
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
        '''Predict a chromosome and sum the overlapping windows on the GPU.

        The rebuilt matrix is banded: a window of W bins places nothing further
        than W-1 off the diagonal, so the whole chromosome is a (W,
        matrix_size) float32 array. Each window contributes to it and is then
        finished with -- which is why shipping the windows themselves to the
        host is so wasteful. Every cell of the band is covered by up to W
        windows, so the per-window triangles are up to W redundant copies of
        the same result: for a genome at 2 kb, 397 GB of them against 3 GB of
        band.

        Summing where the values are produced removes that entirely. Only the
        band crosses the bus, once per chromosome, and no per-window file is
        written at all. For chromosome 1 at 2 kb the band is 248 MB, which fits
        on the card next to the denoiser with room to spare.

        The scatter is a constant index pattern shifted along the chromosome:
        the cell a window's (r, c) entry lands in is ((c-r), r + j), where j is
        the window's start. So the flat indices are one precomputed vector plus
        j, and duplicate targets within a batch are summed by the atomic add.

        `window_starts` are the sliding-window positions the samples came from,
        in the order the dataset yields them, or None when every position was
        predicted and they are simply 0, 1, 2, ... The flanking size is added
        here. Returns the band as (W, matrix_size) float32 together with the
        number of windows summed into it.
        '''
        if self.denoiser is None:
            raise ValueError("denoiser is None. Please call loaddenoiser() first.")
        W = int(window_size)
        matrixSize = int(matrix_size)
        flanking = int(flanking_size)
        starts = None if window_starts is None else np.asarray(window_starts, dtype=np.int32)
        r, c = np.triu_indices(W)
        #where within a window's flattened WxW output each triangle entry sits
        flatWindow = tf.constant((r * W + c).astype(np.int32))
        #and where it lands in the flattened (W, matrix_size) band, before the
        #window's own offset along the chromosome is added
        flatBase = tf.constant(((c - r).astype(np.int64) * matrixSize
                                + r).astype(np.int32))

        band = tf.Variable(tf.zeros([W * matrixSize], dtype=tf.float32),
                           trainable=False)

        @tf.function
        def _step(inp, offsets):
            out = self.predictionStep(input_batch=inp, training=False)
            if out.shape.rank == 4 and out.shape[-1] == 1:
                out = tf.squeeze(out, axis=-1)
            out = tf.reshape(out, [tf.shape(out)[0], W * W])
            #a denoiser trained under mixed precision emits float16; the band
            #is summed in float32 regardless, since up to windowSize windows
            #land on the same cell
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
            #a window at offset j writes columns j .. j+W-1; past the end of the
            #chromosome the flat scatter would wrap into the next diagonal
            #instead of failing, so it is checked rather than trusted
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

    @tf.function
    def predictionStep(self, input_batch, training=False, sample_steps: int = 50):

        rna_condition = input_batch
        if len(rna_condition.shape) == 3:
            rna_condition = tf.expand_dims(rna_condition, axis=-1)
        batch_size = tf.shape(rna_condition)[0]

        x_t = tf.random.normal(shape=(batch_size, self.input_size, self.input_size, 1))

        stride = tf.maximum(1, self.timesteps // sample_steps)

        for t_step in tf.range(self.timesteps, 0, -stride):
            t_batch = tf.fill([batch_size, 1], tf.cast(t_step, tf.float32))
            t_prev = tf.maximum(0, t_step - stride)

            predicted_noise = self.denoiser([x_t, rna_condition, t_batch], training=False)

            # Compute effective alpha and beta across the strided interval (t_prev -> t_step)
            alpha_bar_t = self.alphas_bar[t_step]
            alpha_bar_prev = self.alphas_bar[t_prev]
            alpha_step = alpha_bar_t / alpha_bar_prev
            beta_step = 1.0 - alpha_step

            # Estimate x_0 and clip to [0, 1] to prevent negative/unbounded drift
            x0_pred = (x_t - tf.sqrt(1.0 - alpha_bar_t) * predicted_noise) / tf.sqrt(alpha_bar_t)
            x0_pred = tf.clip_by_value(x0_pred, 0.0, 1.0)

            posterior_mean = (
                (tf.sqrt(alpha_bar_prev) * beta_step / (1.0 - alpha_bar_t)) * x0_pred +
                (tf.sqrt(alpha_step) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * x_t
            )
            posterior_var = beta_step * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

            noise = tf.random.normal(shape=tf.shape(x_t))
            noise_term = tf.where(t_prev > 0, tf.sqrt(posterior_var) * noise, tf.zeros_like(x_t))
            x_t = posterior_mean + noise_term

        # Enforce symmetry and [0, 1] bounds on the final denoised matrix x_0
        x_t = 0.5 * (x_t + tf.transpose(x_t, perm=[0, 2, 1, 3]))
        x_t = tf.clip_by_value(x_t, 0.0, 1.0)
        return x_t

    # def load_denoiser(self, trainedModelPath: str):
    #     '''
    #         load a trained denoiser model for prediction
    #     '''
    #     try:
    #         trainedModel = tf.keras.models.load_model(filepath=trainedModelPath, 
    #                                               custom_objects={"CustomReshapeLayer": CustomReshapeLayer(self.input_size)}, safe_mode=False)
    #         self.denoiser = trainedModel
    #     except Exception as e:
    #         msg = str(e)
    #         msg += "\nError: failed to load trained model"
    #         raise ValueError(msg)
    def load_denoiser(self, trainedModelPath: str):
        '''
        Load a trained denoiser model for prediction with detailed diagnostics.
        '''
        try:
            print(f"Attempting to load model from: {trainedModelPath}")
            
            # 1. Define custom layers
            custom_layers = {
                "CustomReshapeLayer": CustomReshapeLayer,
                "GroupNormalization": keras.layers.GroupNormalization,
                "SpatialTimeEncoding": SpatialTimeEncoding,
            }

            if os.path.isdir(trainedModelPath) or "ckpt" in trainedModelPath:
                ckpt = tf.train.Checkpoint(denoiser=self.denoiser)
                latest = tf.train.latest_checkpoint(trainedModelPath) if os.path.isdir(trainedModelPath) else trainedModelPath
                if latest:
                    print(f"Restoring denoiser from checkpoint: {latest}")
                    ckpt.restore(latest).expect_partial()
                else:
                    raise FileNotFoundError(f"No checkpoint found at {trainedModelPath}")
            else:
                self.denoiser = tf.keras.models.load_model(
                    filepath=trainedModelPath, 
                    custom_objects=custom_layers, 
                    safe_mode=False
                )
            print("Model loaded successfully. Running diagnostics...")

            # --- DIAGNOSTICS ---
            # Check 1: Are weights all zeros or NaNs?
            all_weights = self.denoiser.get_weights()
            if len(all_weights) > 0:
                total_weight_sum = np.sum([np.sum(np.abs(w)) for w in all_weights])
                has_nan = np.any([np.isnan(np.sum(w)) for w in all_weights])
                
                print(f" -> Total sum of weights: {total_weight_sum}")
                
                if has_nan:
                    raise ValueError("CRITICAL: The loaded model contains NaN (Not a Number) weights. Training failed. Decreasing learning rate or increasing GroupNorm epsilon might help.")
                
                if total_weight_sum == 0:
                    raise ValueError("CRITICAL: All model weights are ZERO. The model did not learn or was not saved correctly.")
            
            print(" -> Diagnostics passed. Weights look valid.")

        except Exception as e:
            msg = f"Failed to load denoiser.\nDetail: {str(e)}"
            raise ValueError(msg)

    # Alias for backward compatibility
    loaddenoiser = load_denoiser
        
class CustomReshapeLayer(keras.layers.Layer):
    '''
    reshape a 1D tensor such that it represents 
    the upper triangular part of a square 2D matrix with shape (matsize, matsize)
    #example: 
     [1,2,3,4,5,6] => [[1,2,3],
                       [0,4,5],
                       [0,0,6]]
    '''
    def __init__(self, matsize, **kwargs):
        super(CustomReshapeLayer, self).__init__(**kwargs)
        self.matsize = matsize
        self.triu_indices = [ [x,y] for x,y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1]) ]

    def call(self, inputs):      
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)
        
    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputVec, 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

    def get_config(self):
        return {"matsize": self.matsize}

