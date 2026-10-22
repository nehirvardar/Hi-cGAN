import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU, Flatten, Dense
from tensorflow.keras.layers import GroupNormalization
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import utils
import re

#implementation of adapted pix2pix cGAN
#modified from tensorflow tutorial https://www.tensorflow.org/tutorials/generative/pix
#also see: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/


import logging
log = logging.getLogger(__name__)


class HiCGAN():
    def __init__(self, log_dir: str, 
                    number_factors: int,
                    loss_weight_pixel: float = 100, #factor for L1/L2 loss, like Isola et al. 2017
                    loss_weight_adversarial: float = 1.0, #factor for adversarial loss in generator
                    loss_weight_discriminator: float = 0.5, #factor for disc loss, like Isola et al. 2017
                    loss_weight_tv: float = 1e-10, #factor for TV loss
                    loss_type_pixel: str = "L1", #type of per-pixel error (L1, L2)
                    input_size: int = 256,
                    plot_frequency: int = 20,
                    plot_type: str = "png",
                    learning_rate_generator: float = 2e-5,
                    learning_rate_discriminator: float = 1e-6,
                    adam_beta_1: float = 0.5,
                    pretrained_model_path: str = "",
                    scope=None,
                    mixed_precision: bool = False,
                    restore_checkpoint: bool = False,
                    use_embedding: bool = True,
                    enforce_symmetry: bool = True):
        super().__init__()
        # When True, optimizers are wrapped for loss scaling and the loss
        # functions cast their inputs to float32. The global mixed_float16
        # policy itself is set by the caller (training.main) before this model
        # is constructed so that all layers pick it up.
        self.mixed_precision = mixed_precision
        self.use_embedding = bool(use_embedding)
        self.enforce_symmetry = bool(enforce_symmetry)

        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.input_size = 256
        if input_size in [64,128,256, 512, 768, 1024]:
            self.input_size = input_size
        self.number_factors = number_factors
        self.loss_weight_pixel = loss_weight_pixel
        self.loss_weight_discriminator = loss_weight_discriminator
        self.loss_weight_adversarial = loss_weight_adversarial
        self.loss_weight_tv = loss_weight_tv
        self.loss_type_pixel = loss_type_pixel

        init_msg = (
            f"[MODEL-INIT] HiCGAN configuration: input_size={self.input_size}, number_factors={self.number_factors}, "
            f"use_embedding={self.use_embedding} ({'1D BigWig -> cnn_embedding' if self.use_embedding else 'Direct 2D Input'}), "
            f"enforce_symmetry={self.enforce_symmetry}, loss_type_pixel={self.loss_type_pixel}, "
            f"mixed_precision={self.mixed_precision}"
        )
        log.info(init_msg)
        print(init_msg)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator, beta_1=adam_beta_1, name="Adam_Generator")
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator, beta_1=adam_beta_1, name="Adam_Discriminator")
        if self.mixed_precision:
            # Dynamic loss scaling to keep float16 gradients from underflowing.
            self.generator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.generator_optimizer)
            self.discriminator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.discriminator_optimizer)

        if self.use_embedding:
            self.generator_embedding = self.cnn_embedding()
            self.discriminator_embedding = self.cnn_embedding()
        else:
            self.generator_embedding = None
            self.discriminator_embedding = None
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.log_dir=log_dir
        
        self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)
        self.start_epoch = 0
        if restore_checkpoint:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            print(f"Current checkpoint directory: {self.checkpoint_dir}")
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
                # self.__epoch_counter = ckpt_num
                self.start_epoch = ckpt_num
                print(f"Resuming from checkpoint #{ckpt_num}.")
            else:
                print("No checkpoint found. Starting from scratch.")
        self.plot_type = plot_type
        if self.plot_type not in ["png", "pdf", "svg"]:
            self.plot_type = "png"
            print("plot type {:s} unsupported, changed to png".format(plot_type))
        self.progress_plot_name = os.path.join(self.log_dir, "lossOverEpochs.{:s}".format(self.plot_type))
        self.progress_plot_frequency = plot_frequency
        self.example_plot_frequency = 5

        #losses per epochs
        self.__gen_train_loss_epochs = []
        self.__disc_train_loss_true_epochs = []
        self.__disc_train_loss_fake_epochs = []
        self.__gen_val_loss_epochs = []
        self.__disc_val_loss_epochs = []
        #losses per batch
        self.__gen_train_loss_batches = []
        self.__disc_train_loss_true_batches = []
        self.__disc_train_loss_fake_batches = []
        self.__gen_val_loss_batches = []
        self.__disc_val_loss_batches = []

        self.__epoch_counter = 0
        self.__batch_counter = 0

        self.scope = scope

    def cnn_embedding(self, nr_filters_list=[1024,512,512,256,256,128,128,64], kernel_width_list=[4,4,4,4,4,4,4,4], apply_dropout: bool = False):  
        inputs = tf.keras.layers.Input(shape=(3*self.input_size, self.number_factors))
        #add 1D convolutions
        x = inputs
        for i, (nr_filters, kernelWidth) in enumerate(zip(nr_filters_list, kernel_width_list)):
            convParamDict = dict()
            convParamDict["name"] = "conv1D_" + str(i + 1)
            convParamDict["filters"] = nr_filters
            convParamDict["kernel_size"] = kernelWidth
            convParamDict["data_format"]="channels_last"
            convParamDict["kernel_regularizer"]=tf.keras.regularizers.l2(0.01)
            if kernelWidth > 1:
                convParamDict["padding"] = "same"
            x = Conv1D(**convParamDict)(x)
            # x = BatchNormalization()(x)
            x = GroupNormalization(groups=nr_filters, axis=-1)(x)
            if apply_dropout:
                x = Dropout(0.5)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        #make the shape of a square matrix
        x = Conv1D(filters=self.input_size, 
                    strides=3, 
                    kernel_size=4, 
                    data_format="channels_last", 
                    activation="sigmoid", 
                    padding="same", name="conv1D_final")(x)
        #ensure the matrix is symmetric, i.e. x = transpose(x)
        if self.enforce_symmetry:
            x_T = tf.keras.layers.Permute((2,1))(x) #this is the matrix transpose
            x = tf.keras.layers.Add()([x, x_T])
            x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x) #add transpose and divide by 2
        #reshape the matrix into a 2D grayscale image
        x = tf.keras.layers.Reshape((self.input_size,self.input_size,self.INPUT_CHANNELS))(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name="CNN-embedding")
        #model.build(input_shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        #model.summary()
        return model


    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            # result.add(BatchNormalization())
            result.add(GroupNormalization(groups=filters, axis=-1))
        result.add(LeakyReLU(alpha=0.2))
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
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


    def Generator(self):
        if self.use_embedding:
            inputs = tf.keras.layers.Input(shape=[3*self.input_size, self.number_factors], name="factorData")
            x = self.generator_embedding(inputs)
        else:
            inputs = tf.keras.layers.Input(shape=[self.input_size, self.input_size, self.INPUT_CHANNELS], name="factorData")
            x = inputs

        #the downsampling part of the network, defined for 256x256 images
        down_stack = [
            HiCGAN.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            HiCGAN.downsample(128, 4), # (bs, 64, 64, 128)
            HiCGAN.downsample(256, 4), # (bs, 32, 32, 256)
            HiCGAN.downsample(512, 4), # (bs, 16, 16, 512)
            HiCGAN.downsample(512, 4), # (bs, 8, 8, 512)
            HiCGAN.downsample(512, 4), # (bs, 4, 4, 512)
            HiCGAN.downsample(512, 4), # (bs, 2, 2, 512)
            HiCGAN.downsample(512, 4, apply_batchnorm=False), # (bs, 1, 1, 512)
        ]
        #if the input images are smaller, leave out some layers accordingly
        if self.input_size < 256:
            down_stack = down_stack[:-2] + down_stack[-1:]
        if self.input_size < 128:
            down_stack = down_stack[:-2] + down_stack[-1:]

        #the upsampling portion of the generator, designed for 256x256 images
        up_stack = [
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            HiCGAN.upsample(512, 4), # (bs, 16, 16, 1024)
            HiCGAN.upsample(256, 4), # (bs, 32, 32, 512)
            HiCGAN.upsample(128, 4), # (bs, 64, 64, 256)
            HiCGAN.upsample(64, 4), # (bs, 128, 128, 128)
        ]
        #for smaller images, take layers away, otherwise downsampling won't work
        if self.input_size < 256:
            up_stack = up_stack[:2] + up_stack[3:]
        if self.input_size < 128:
            up_stack = up_stack[:2] + up_stack[3:]

        initializer = tf.random_normal_initializer(0., 0.02)

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer) 
        x = last(x)
        #enforce symmetry only in diagonal window mode
        if self.enforce_symmetry:
            x_T = tf.keras.layers.Permute((2,1,3))(x)
            x = tf.keras.layers.Add()([x, x_T])
            x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x)
        x = tf.keras.layers.Activation("sigmoid")(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="Generator")


    def generator_loss(self, disc_generated_output, gen_output, target):
        # Compute the loss in float32 (no-op without mixed precision; required
        # for numerical stability and dtype matching when float16 is active).
        disc_generated_output = tf.cast(disc_generated_output, tf.float32)
        gen_output = tf.cast(gen_output, tf.float32)
        target = tf.cast(target, tf.float32)
        advers_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_generated_output), logits=disc_generated_output) )
        # mean squared error or mean absolute error
        if self.loss_type_pixel == "L1":
            pixel_loss = tf.reduce_mean(tf.abs(target - gen_output))
        else: 
            pixel_loss = tf.reduce_mean(tf.square(target - gen_output))
        tv_loss = tf.reduce_mean(tf.image.total_variation(gen_output))
        total_gen_loss = self.loss_weight_pixel * pixel_loss + self.loss_weight_adversarial * advers_loss + self.loss_weight_tv * tv_loss
        return total_gen_loss, advers_loss, pixel_loss


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        def _maybe_symmetrize(tensor):
            if not self.enforce_symmetry:
                return tensor
            t_T = tf.keras.layers.Permute((2, 1, 3))(tensor)
            t = tf.keras.layers.Add()([tensor, t_T])
            return tf.keras.layers.Lambda(lambda z: 0.5 * z)(t)

        if self.use_embedding:
            inp = tf.keras.layers.Input(shape=[3*self.input_size, self.number_factors], name='input_image')
            d = self.discriminator_embedding(inp)
        else:
            inp = tf.keras.layers.Input(shape=[self.input_size, self.input_size, self.INPUT_CHANNELS], name='input_image')
            d = inp

        tar = tf.keras.layers.Input(shape=[self.input_size, self.input_size, self.OUTPUT_CHANNELS], name='target_image')
        #Patch-GAN (Isola et al.)
        d = tf.keras.layers.Concatenate()([d, tar])
        if self.input_size > 64:
            #downsample and symmetrize 1 
            d = HiCGAN.downsample(64, 4, False)(d) # (bs, inp.size/2, inp.size/2, 64)
            d = _maybe_symmetrize(d)
            #downsample and symmetrize 2
            d = HiCGAN.downsample(128, 4)(d)# (bs, inp.size/4, inp.size/4, 128)
            d = _maybe_symmetrize(d)
        else:    
            #downsample and symmetrize 3
            d = HiCGAN.downsample(256, 4)(d)
            d = _maybe_symmetrize(d)
        #downsample and symmetrize 4
        d = HiCGAN.downsample(256, 4)(d) # (bs, inp.size/8, inp.size/8, 256)
        d = _maybe_symmetrize(d)
        d = Conv2D(512, 4, strides=1, padding="same", kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 512)
        d = _maybe_symmetrize(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, 4, strides=1, padding="same",
                        kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 1)
        d = _maybe_symmetrize(d)
        #d = tf.keras.layers.Activation("sigmoid")(d) #sigmoid will be done in the loss function itself
        return tf.keras.Model(inputs=[inp, tar], outputs=d, name="Discriminator")

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        disc_real_output = tf.cast(disc_real_output, tf.float32)
        disc_generated_output = tf.cast(disc_generated_output, tf.float32)
        real_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output), logits=disc_real_output) )
        generated_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output) )
        total_disc_loss = self.loss_weight_discriminator * (real_loss + generated_loss)
        return total_disc_loss, real_loss, generated_loss

    @tf.function
    def distributed_train_step(self, input_data):
        input_image, target = input_data[0]["factorData"], input_data[1]["out_matrixData"]
        per_replica_losses = self.scope.run(self.train_step, args=(input_image, target, ))
        return self.scope.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss, disc_real_loss, disc_gen_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            if self.mixed_precision:
                # scaled loss must be computed inside the tape
                scaled_gen_loss = self.generator_optimizer.get_scaled_loss(gen_total_loss)
                scaled_disc_loss = self.discriminator_optimizer.get_scaled_loss(disc_loss)

        if self.mixed_precision:
            generator_gradients = self.generator_optimizer.get_unscaled_gradients(
                gen_tape.gradient(scaled_gen_loss, self.generator.trainable_variables))
            discriminator_gradients = self.discriminator_optimizer.get_unscaled_gradients(
                disc_tape.gradient(scaled_disc_loss, self.discriminator.trainable_variables))
        else:
            generator_gradients = gen_tape.gradient(gen_total_loss,
                                                    self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))
        return gen_total_loss, disc_loss, disc_real_loss, disc_gen_loss

    @tf.function
    def validationStep(self, input_image, target):
        gen_output = self.generator(input_image, training=True)

        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

        gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss, _, _ = self.discriminator_loss(disc_real_output, disc_generated_output)
        return gen_total_loss, disc_loss

    def generate_images(self, model, test_input, tar, epoch: int):
        inp_tensor = test_input["factorData"] if isinstance(test_input, dict) else test_input
        prediction = model(inp_tensor, training=True)
        tar_img = tf.cast(tar["out_matrixData"][0], tf.float32)
        pred_img = tf.cast(prediction[0], tf.float32)
        pred_mse = float(tf.reduce_mean(tf.square(tar_img - pred_img)))
        figname = "testpred_epoch_{:05d}.png".format(epoch)
        figname = os.path.join(self.log_dir, figname)
        inp_sample = np.asarray(inp_tensor[0])
        if inp_sample.ndim == 3 and inp_sample.shape[-1] == 1:
            inp_sample = inp_sample[:, :, 0]
        display_list = [inp_sample, np.squeeze(np.asarray(tar_img)), np.squeeze(np.asarray(pred_img))]
        titleList = ['Input Feature / 2D Random', 'Ground Truth', 'Predicted Image (MSE: {:.5f})'.format(pred_mse)]

        fig1, axs1 = plt.subplots(1,len(display_list), figsize=(15,15))
        for i in range(len(display_list)):
            axs1[i].imshow(display_list[i] * 0.5 + 0.5, aspect='auto')
            axs1[i].set_title(titleList[i])
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

        for epoch in range(self.start_epoch, epochs):
            #generate sample output
            if epoch % self.example_plot_frequency == 0:
                for example_input, example_target in test_ds.take(1):
                    self.generate_images(self.generator, example_input, example_target, epoch)
            
            
            train_pbar = tqdm(total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            train_samples_in_epoch = 0

            for distributed_input in distributed_dataset:
               
                train_samples_in_epoch += 1
                gen_loss, _, disc_loss_real, disc_loss_fake = self.distributed_train_step(distributed_input)

                self.__disc_train_loss_true_batches.append(disc_loss_real)
                self.__disc_train_loss_fake_batches.append(disc_loss_fake)
                self.__gen_train_loss_batches.append(gen_loss)
                train_bar_postfixDict= {"g": "{:.4f}".format(self.__gen_train_loss_batches[-1]),
                                         "dt": "{:.3f}".format(self.__disc_train_loss_true_batches[-1]),
                                         "df": "{:.3f}".format(self.__disc_train_loss_fake_batches[-1])}
                if len(self.__gen_val_loss_epochs) > 0:
                    train_bar_postfixDict["v"] = "{:.3f}".format(self.__gen_val_loss_epochs[-1])
                train_pbar.set_postfix( train_bar_postfixDict )
                train_pbar.update(1)
                self.__batch_counter += 1
            train_pbar.close()
            self.__gen_train_loss_epochs.append(np.mean(self.__gen_train_loss_batches[-train_samples_in_epoch:]))
            self.__disc_train_loss_true_epochs.append(np.mean(self.__disc_train_loss_true_batches[-train_samples_in_epoch:]))
            self.__disc_train_loss_fake_epochs.append(np.mean(self.__disc_train_loss_fake_batches[-train_samples_in_epoch:]))

            # Validation
            validation_samples_in_epoch = 0
            vds = test_ds.take(validation_steps) if validation_steps else test_ds
            for input_image, target in vds:
                gen_loss_val, disc_loss_val = self.validationStep(input_image["factorData"], target["out_matrixData"])
                self.__gen_val_loss_batches.append(gen_loss_val)
                self.__disc_val_loss_batches.append(disc_loss_val)
                validation_samples_in_epoch += 1
            self.__disc_val_loss_epochs.append(np.mean(self.__disc_val_loss_batches[-validation_samples_in_epoch:]))
            self.__gen_val_loss_epochs.append(np.mean(self.__gen_val_loss_batches[-validation_samples_in_epoch:]))

            #count the epoch
            self.__epoch_counter += 1

            # saving the model every 20 epochs
            if (epoch + 1) % self.progress_plot_frequency == 0:
                #plot loss
                utils.plotLoss(pGeneratorLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                              pDiscLossValueLists=[ [self.loss_weight_discriminator*sum(x) for x in zip(self.__disc_train_loss_fake_epochs, self.__disc_train_loss_true_epochs)],
                                                    self.__disc_train_loss_true_epochs, 
                                                    self.__disc_train_loss_fake_epochs, 
                                                    self.__disc_val_loss_epochs],
                              pGeneratorLossNameList=["training", "validation"],
                              pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
                np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=self.__gen_train_loss_batches, 
                                    genLossVal=self.__gen_val_loss_batches, 
                                    discLossTrain_True=self.__disc_train_loss_true_batches,
                                    discLossTrain_Fake=self.__disc_train_loss_fake_batches, 
                                    discLossVal=self.__disc_val_loss_batches)
                self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.keras".format(epoch)), save_format="keras")
                self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.keras".format(epoch)), save_format="keras")


            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        utils.plotLoss(pGeneratorLossValueLists=[self.__gen_train_loss_epochs, self.__gen_val_loss_epochs],
                              pDiscLossValueLists=[ [self.loss_weight_discriminator*sum(x) for x in zip(self.__disc_train_loss_fake_epochs, self.__disc_train_loss_true_epochs)],
                                                    self.__disc_train_loss_true_epochs, 
                                                    self.__disc_train_loss_fake_epochs, 
                                                    self.__disc_val_loss_epochs],
                              pGeneratorLossNameList=["training", "validation"],
                              pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
        np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=self.__gen_train_loss_batches, 
                                    genLossVal=self.__gen_val_loss_batches, 
                                    discLossTrain_True=self.__disc_train_loss_true_batches,
                                    discLossTrain_Fake=self.__disc_train_loss_fake_batches, 
                                    discLossVal=self.__disc_val_loss_batches)
        self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.keras".format(epoch)), save_format="keras")
        self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.keras".format(epoch)), save_format="keras")

    def plotModels(self, pOutputPath: str, pFigureFileFormat: str):
        generatorPlotName = "generatorModel.{:s}".format(pFigureFileFormat)
        generatorPlotName = os.path.join(pOutputPath, generatorPlotName)
        discriminatorPlotName = "discriminatorModel.{:s}".format(pFigureFileFormat)
        discriminatorPlotName = os.path.join(pOutputPath, discriminatorPlotName)
        generatorEmbeddingPlotName = "generatorEmbeddingModel.{:s}".format(pFigureFileFormat)
        generatorEmbeddingPlotName = os.path.join(pOutputPath, generatorEmbeddingPlotName)
        discriminatorEmbeddingPlotName = "discriminatorEmbeddingModel.{:s}".format(pFigureFileFormat)
        discriminatorEmbeddingPlotName = os.path.join(pOutputPath, discriminatorEmbeddingPlotName)
        #Architecture diagrams are a nicety, and they need pydot plus a graphviz
        #binary. Without them keras raises ImportError, which used to abort the
        #whole training run before a single step -- losing the job over a
        #picture. Warn and carry on instead.
        modelsToPlot = [
            (self.generator, generatorPlotName),
            (self.discriminator, discriminatorPlotName),
        ]
        if self.use_embedding and self.generator_embedding is not None and self.discriminator_embedding is not None:
            modelsToPlot.extend([
                (self.generator_embedding, generatorEmbeddingPlotName),
                (self.discriminator_embedding, discriminatorEmbeddingPlotName),
            ])
        for model, name in modelsToPlot:
            try:
                tf.keras.utils.plot_model(model, show_shapes=True, to_file=name)
            except (ImportError, OSError) as ex:
                msg = ("skipping the architecture diagram {:s}: {:s}. "
                       "Install pydot and graphviz if you want it; training is "
                       "unaffected.").format(name, str(ex).split("\n")[0])
                print(msg)
                break

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

        if self.generator is None:
            raise ValueError("Generator is None. Please call loadGenerator() first.")

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
        if self.generator is None:
            raise ValueError("Generator is None. Please call loadGenerator() first.")
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
        if self.generator is None:
            raise ValueError("Generator is None. Please call loadGenerator() first.")
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
        on the card next to the generator with room to spare.

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
        if self.generator is None:
            raise ValueError("Generator is None. Please call loadGenerator() first.")
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
            #a generator trained under mixed precision emits float16; the band
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
    def predictionStep(self, input_batch, training=False):
        return self.generator(input_batch, training=training)
  
    
    # def loadGenerator(self, trainedModelPath: str):
    #     '''
    #         load a trained generator model for prediction
    #     '''
    #     try:
    #         trainedModel = tf.keras.models.load_model(filepath=trainedModelPath, 
    #                                               custom_objects={"CustomReshapeLayer": CustomReshapeLayer(self.input_size)}, safe_mode=False)
    #         self.generator = trainedModel
    #     except Exception as e:
    #         msg = str(e)
    #         msg += "\nError: failed to load trained model"
    #         raise ValueError(msg)
    def loadGenerator(self, trainedModelPath: str):
        '''
        Load a trained generator model for prediction with detailed diagnostics.
        '''
        try:
            print(f"Attempting to load model from: {trainedModelPath}")
            
            # 1. Define custom layers
            custom_layers = {
                "CustomReshapeLayer": CustomReshapeLayer,
            }
            
            # 2. Add GroupNormalization
            # Try native Keras (TF 2.12+)
            # if hasa?ttr(tf.keras.layers, "GroupNormalization"):
            custom_layers["GroupNormalization"] = tf.keras.layers.GroupNormalization
            # else:
            #     # Fallback for older TF versions using addons
            #     try:
            #         import tensorflow_addons as tfa
            #         custom_layers["GroupNormalization"] = tfa.layers.GroupNormalization
            #     except ImportError:
            #         print("Warning: Could not import GroupNormalization from keras or addons.")

            # 3. Load the model
            self.generator = tf.keras.models.load_model(
                filepath=trainedModelPath, 
                custom_objects=custom_layers, 
                safe_mode=False
            )
            print("Model loaded successfully. Running diagnostics...")

            # --- DIAGNOSTICS ---
            # Check 1: Are weights all zeros or NaNs?
            all_weights = self.generator.get_weights()
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
            msg = f"Failed to load generator.\nDetail: {str(e)}"
            raise ValueError(msg)
        
class CustomReshapeLayer(tf.keras.layers.Layer):
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