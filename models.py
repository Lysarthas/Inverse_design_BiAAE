import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, enc, dec, latent_dim, lambda_rec):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.z_dim = latent_dim
        self.lambda_r = lambda_rec
        
    def compile(self, rec_obj, rec_opt):
        super(VAE, self).compile()
        self.rec_loss = rec_obj
        self.optimizer = rec_opt

    def summary(self):
        self.enc.summary()
        self.dec.summary()

    @staticmethod
    def sample_z(mean, logvar):
        return mean + tf.random.normal(tf.shape(mean)) * tf.math.exp(0.5*logvar)
    
    @staticmethod
    def kl_div(mean, logvar):
        return -0.5 * tf.reduce_sum(
            tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
        )

    def train_step(self, batch_data):
        x = batch_data

        with tf.GradientTape() as ae_tape:
            z = self.enc(x, training=True)
            z_mean, z_logvar = tf.split(z, [self.z_dim, self.z_dim], -1)
            z_enc = self.sample_z(z_mean, z_logvar)
            
            l_kl = self.kl_div(z_mean, z_logvar)
            l_rec = tf.reduce_mean(self.rec_loss(x, self.dec(z_enc, training=True)))
            l_total = self.lambda_r * l_rec + l_kl

        grads = ae_tape.gradient(l_total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            'loss': l_total,
            'rec_loss': l_rec,
            'kl_loss': l_kl
        }





class BiAAE(tf.keras.Model):
    def __init__(self, seq_enc, seq_dec, img_enc, img_dec, disc, disc_indp_x, disc_indp_y, z_dim, joint_dim, dis_step, rec_lamb, joint_lamb, norm_lamb, x_indp_lamb, y_indp_lamb):
        super(BiAAE, self).__init__()
        self.seqenc = seq_enc
        self.seqdec = seq_dec
        self.imgenc = img_enc
        self.imgdec = img_dec
        self.dis = disc
        self.dis_x = disc_indp_x
        self.dis_y = disc_indp_y
        self.z_dim = z_dim
        self.j_dim = joint_dim
        self.dis_step = dis_step
        # self.gen_step = gen_step
        self.rec_lamb = rec_lamb
        self.joint_lamb = joint_lamb
        self.norm_lamb = norm_lamb
        self.x_lamb = x_indp_lamb
        self.y_lamb = y_indp_lamb

    def compile(self, ts_rec_obj, im_rec_obj, dis_opt, dis_obj, gen_opt, gen_obj):
        super(BiAAE, self).compile()
        self.dis_optimizer = dis_opt
        self.gen_optimizer = gen_opt
        self.rec_loss_ts = ts_rec_obj
        self.rec_loss_im = im_rec_obj
        self.dis_loss_fn = dis_obj
        self.gen_loss_fn = gen_obj

    def summary(self):
        self.seqenc.summary()
        self.seqdec.summary()
        self.imgenc.summary()
        self.imgdec.summary()
        self.dis.summary()

    def train_step(self, batch_data):
        seq_x, img_x = batch_data
        batch_size = tf.shape(seq_x)[0]

        with tf.GradientTape() as gen_tape:
            latent_seq = self.seqenc(seq_x, training=True)
            latent_img = self.imgenc(img_x, training=True)
            seq_z, seq_j = tf.split(latent_seq, [self.z_dim-self.j_dim, self.j_dim], -1)
            img_z, img_j = tf.split(latent_img, [self.z_dim-self.j_dim, self.j_dim], -1)

            loss_rec_seq = (tf.reduce_mean(self.rec_loss_ts(seq_x, self.seqdec(latent_seq, training=True))) + 
                            tf.reduce_mean(self.rec_loss_ts(seq_x, self.seqdec(tf.concat([seq_z, img_j], axis=1), training=True))))/2.0

            loss_rec_img = (tf.reduce_mean(self.rec_loss_im(img_x, self.imgdec(latent_img, training=True))) +
                            tf.reduce_mean(self.rec_loss_im(img_x, self.imgdec(tf.concat([img_z, seq_j], axis=1), training=True))))/2.0
            
            loss_joint = tf.reduce_mean(tf.norm(seq_j-img_j, 2, -1))

            disc_out = self.dis(
                tf.concat([tf.concat([seq_z, seq_j, img_z], axis=-1), tf.concat([seq_z, img_j, img_z], axis=-1)],axis=0)
                )
            loss_norm = tf.reduce_mean(self.dis_loss_fn(disc_out, tf.ones_like(disc_out)))

            disc_x_out = self.dis_x(
                tf.concat([seq_z, img_j, img_z], axis=-1)
            )
            loss_x = tf.reduce_mean(self.dis_loss_fn(disc_x_out, tf.ones_like(disc_x_out)))

            disc_y_out = self.dis_y(
                tf.concat([seq_z, seq_j, img_z], axis=-1)
            )
            loss_y = tf.reduce_mean(self.dis_loss_fn(disc_y_out, tf.ones_like(disc_y_out)))

            gen_loss = ((loss_rec_seq + loss_rec_img)*self.rec_lamb + 
                        loss_joint*self.joint_lamb + 
                        loss_norm*self.norm_lamb + 
                        loss_x*self.x_lamb + 
                        loss_y*self.y_lamb
                        )
            trainable_var_gen = (self.seqenc.trainable_variables + 
                                 self.seqdec.trainable_variables + 
                                 self.imgenc.trainable_variables +
                                 self.imgdec.trainable_variables
                                )
            gen_grad = gen_tape.gradient(gen_loss, trainable_var_gen)
            self.gen_optimizer.apply_gradients(
                zip(gen_grad, trainable_var_gen)
            )

        for _ in range(self.dis_step):
            with tf.GradientTape() as dis_tape:
                latent_seq = self.seqenc(seq_x)
                latent_img = self.imgenc(img_x)
                seq_z, seq_j = tf.split(latent_seq, [self.z_dim-self.j_dim, self.j_dim], -1)
                img_z, img_j = tf.split(latent_img, [self.z_dim-self.j_dim, self.j_dim], -1)

                fake_norm = tf.concat([tf.concat([seq_z, seq_j, img_z], axis=-1), tf.concat([seq_z, img_j, img_z], axis=-1)], axis=0)
                dis_fake = self.dis(fake_norm, training=True)
                real_norm_dist = tf.random.normal(tf.shape(fake_norm), 0.0, 1.0)
                dis_real = self.dis(real_norm_dist, training=True)
                dis_res = tf.concat([dis_fake, dis_real], axis=0)
                target = tf.concat([tf.zeros_like(dis_fake), tf.ones_like(dis_real)], axis=0)
                loss_dis_norm = tf.reduce_mean(self.dis_loss_fn(dis_res, target))

                fake_x = tf.concat([seq_z, seq_j, img_z], axis=-1)
                dis_x_fake = self.dis_x(fake_x, training=True)
                real_x = tf.concat([tf.gather(seq_z, tf.random.shuffle(tf.range(batch_size))), seq_j, img_z], axis=-1)
                dis_x_real = self.dis_x(real_x, training=True)
                dis_x_res = tf.concat([dis_x_fake, dis_x_real], axis=0)
                target_x = tf.concat([tf.zeros_like(dis_x_fake), tf.ones_like(dis_x_real)], axis=0)
                loss_dis_x = tf.reduce_mean(self.dis_loss_fn(dis_x_res, target_x))

                fake_y = tf.concat([seq_z, img_j, img_z], axis=-1)
                dis_y_fake = self.dis_y(fake_y, training=True)
                real_y = tf.concat([seq_z, img_j, tf.gather(img_z, tf.random.shuffle(tf.range(batch_size)))], axis=-1)
                dis_y_real = self.dis_y(real_y, training=True)
                dis_y_res = tf.concat([dis_y_fake, dis_y_real], axis=0)
                target_y = tf.concat([tf.zeros_like(dis_y_fake), tf.ones_like(dis_y_real)], axis=0)
                loss_dis_y = tf.reduce_mean(self.dis_loss_fn(dis_y_res, target_y))

                dis_loss = (loss_dis_norm + loss_dis_x + loss_dis_y)
                trainable_var_dis = (self.dis.trainable_variables + self.dis_x.trainable_variables + self.dis_y.trainable_variables)

                dis_grad = dis_tape.gradient(dis_loss, trainable_var_dis)
                self.dis_optimizer.apply_gradients(
                    zip(dis_grad, trainable_var_dis)
                )
        
        return {
            'gen_loss': gen_loss,
            'dis_loss': dis_loss
        }


    
class CRISPAAE(tf.keras.Model):
    def __init__(self, seq_enc, seq_dec, img_enc, img_dec, disc, scale, z_dim, dis_step, rec_lamb, joint_lamb, dis_lamb):
        super(CRISPAAE, self).__init__()
        self.seqenc = seq_enc
        self.seqdec = seq_dec
        self.imgenc = img_enc
        self.imgdec = img_dec
        self.disc = disc
        self.scale = scale
        self.z_dim = z_dim
        self.dis_step = dis_step
        self.rec_lamb = rec_lamb
        self.joint_lamb = joint_lamb
        self.dis_lamb = dis_lamb

    def compile(self, ts_rec_obj, im_rec_obj, con_obj, dis_opt, dis_obj, gen_opt, gen_obj):
        super(CRISPAAE, self).compile()
        self.dis_optimizer = dis_opt
        self.gen_optimizer = gen_opt
        self.rec_loss_ts = ts_rec_obj
        self.rec_loss_im = im_rec_obj
        self.dis_loss_fn = dis_obj
        self.gen_loss_fn = gen_obj
        self.con_loss_fn = con_obj

    def summary(self):
        self.seqenc.summary()
        self.seqdec.summary()
        self.imgenc.summary()
        self.imgdec.summary()
        self.disc.summary()

    def train_step(self, batch_data):
        seq_x, img_x = batch_data

        with tf.GradientTape() as gen_tape:
            latent_seq = self.seqenc(seq_x, training=True)
            latent_img = self.imgenc(img_x, training=True)

            loss_rec_seq = tf.reduce_mean(self.rec_loss_ts(seq_x, self.seqdec(latent_seq, training=True)))
            loss_rec_img = tf.reduce_mean(self.rec_loss_im(img_x, self.imgdec(latent_img, training=True)))
            
            logits_scale = self.scale
            img_feat = latent_img / tf.norm(latent_img, 2, -1, keepdims=True)
            seq_feat = latent_seq / tf.norm(latent_seq, 2, -1, keepdims=True)
            log_scale = tf.math.exp(logits_scale)
            logits_per_img = log_scale * img_feat @ tf.transpose(seq_feat)
            logits_per_seq = tf.transpose(logits_per_img)
            real_label = tf.range(tf.shape(logits_per_img)[0])
            loss_contrastive = (tf.reduce_mean(self.con_loss_fn(real_label, logits_per_img))+
                                tf.reduce_mean(self.con_loss_fn(real_label, logits_per_seq)))/2.0
            
            fake_dist = tf.concat([latent_img, latent_seq], axis=-1)
            dis_out = self.disc(fake_dist)
            loss_dis = tf.reduce_mean(self.dis_loss_fn(dis_out, tf.ones_like(dis_out)))

            gen_loss = ((loss_rec_seq + loss_rec_img)*self.rec_lamb + 
                        loss_contrastive*self.joint_lamb + 
                        loss_dis*self.dis_lamb
            )

            trainable_var_gen = (self.seqenc.trainable_variables + 
                                 self.seqdec.trainable_variables + 
                                 self.imgenc.trainable_variables +
                                 self.imgdec.trainable_variables +
                                 [logits_scale]
            )
            gen_grad = gen_tape.gradient(gen_loss, trainable_var_gen)
            self.gen_optimizer.apply_gradients(
                zip(gen_grad, trainable_var_gen)
            )

        for _ in range(self.dis_step):
            with tf.GradientTape() as dis_tape:
                latent_seq = self.seqenc(seq_x)
                latent_img = self.imgenc(img_x)

                fake_norm = tf.concat([latent_img, latent_seq], axis=-1)
                real_norm = tf.random.normal(tf.shape(fake_norm), 0, 1)
                dis_fake_out = self.disc(fake_norm, training=True)
                dis_real_out = self.disc(real_norm, training=True)
                l_fake = tf.reduce_mean(self.dis_loss_fn(dis_fake_out, tf.zeros_like(dis_fake_out)))
                l_real = tf.reduce_mean(self.dis_loss_fn(dis_real_out, tf.ones_like(dis_real_out)))

                dis_loss = (l_fake + l_real)/2.0

                trainable_var_dis = (self.disc.trainable_variables)

                dis_grad = dis_tape.gradient(dis_loss, trainable_var_dis)
                self.dis_optimizer.apply_gradients(
                    zip(dis_grad, trainable_var_dis)
                )

        return {
            'gen_loss': gen_loss,
            'dis_loss': dis_loss
        }

