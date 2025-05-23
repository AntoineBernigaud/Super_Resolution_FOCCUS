import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import generic_utils
from meta import input_shapes, Nontrainable


class GAN(object):
    def __init__(self, gen, disc, lr_gen=0.0001, lr_disc=0.0001):
        self.gen = gen
        self.disc = disc
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.build()

    def build(self):
        img_shape = input_shapes(self.disc, "sample_in")[0]
        cond_shapes = input_shapes(self.gen, "cond_in")
        noise_shapes = input_shapes(self.gen, "noise_in")
        mask_shapes = input_shapes(self.disc, "mask_in")

        #print('shapes :',img_shape,cond_shapes,noise_shapes)
        # Create optimizers
        self.opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)

        # Build discriminator training network
        with Nontrainable(self.gen):
            real_image = Input(shape=img_shape)
            cond = [Input(shape=s) for s in cond_shapes]
            noise = [Input(shape=s) for s in noise_shapes]
            mask = [Input(shape=s) for s in mask_shapes]

            disc_real = self.disc([real_image]+cond+mask)
            generated_image = self.gen([cond]+[noise])
            disc_fake = self.disc([generated_image]+cond+mask)

            self.disc_trainer = Model(
                inputs=[real_image]+cond+mask+noise,
                outputs=[disc_real, disc_fake]
            )
            self.disc_trainer.compile(optimizer=self.opt_disc,
                loss=["binary_crossentropy", "binary_crossentropy"])

        # Build generator training network
        with Nontrainable(self.disc):
            cond = [Input(shape=s) for s in cond_shapes]
            noise = [Input(shape=s) for s in noise_shapes]
            mask = [Input(shape=s) for s in mask_shapes]

            generated_image = self.gen(cond+noise)
            disc_fake = self.disc([generated_image]+cond+mask)
            
            self.gen_trainer = Model(
                inputs=cond+mask+noise, 
                outputs=disc_fake
            )
            self.gen_trainer.compile(optimizer=self.opt_gen,
                loss="binary_crossentropy")

    def fit_generator(self, batch_gen, noise_gen, steps_per_epoch=1, num_epochs=1,
        training_disc_per_step=1, training_gan_per_step=1):

        disc_out_shape = (batch_gen.batch_size, self.disc.output_shape[1])
        fake_target = np.ones(disc_out_shape, dtype=np.float32)
        real_target = np.zeros(disc_out_shape, dtype=np.float32)

        for epoch in range(num_epochs):

            print("Epoch {}/{}".format(epoch+1, num_epochs))
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                steps_per_epoch*batch_gen.batch_size*(training_disc_per_step+training_gan_per_step))
            for step in range(steps_per_epoch):

                # Train discriminator
                with Nontrainable(self.gen):
                    for repeat in range(training_disc_per_step):
                        (image_batch,cond_batch,mask_batch) = next(batch_gen)
                        noise_batch = next(noise_gen)
                        disc_loss = self.disc_trainer.train_on_batch(
                            [image_batch,cond_batch,mask_batch] + noise_batch,
                            [real_target, fake_target]
                        )

                # Train generator
                with Nontrainable(self.disc):
                    for repeat in range(training_gan_per_step):
                        (image_batch,cond_batch,mask_batch) = next(batch_gen)
                        noise_batch = next(noise_gen)
                        gen_loss = self.gen_trainer.train_on_batch(
                            [cond_batch] + [mask_batch] + noise_batch, real_target
                        )

                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                losses.append(("G0", gen_loss))
                progbar.add((training_disc_per_step+training_gan_per_step)*batch_gen.batch_size, 
                    values=losses)