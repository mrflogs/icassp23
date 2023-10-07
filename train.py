from __future__ import print_function
import os
import random
import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor, CSRTensor, COOTensor
import numpy as np
import math
import sys
from sklearn import preprocessing
import model
import util
from config import opt
from mindspore import load_checkpoint, load_param_into_net

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

def sample():
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    batch_feature, batch_label, batch_att = data.next_batch_unpair_test(opt.batch_size)
    input_res_unpair.copy_(batch_feature)
    input_att_unpair.copy_(batch_att)

def loss_fn(recon_x, x, mean, log_var):
    BCE = ops.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * ops.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = ops.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    
    if opt.cuda:
        interpolates = interpolates.cuda()
        
    interpolates = Variable(interpolates, requires_grad=True)
    
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = ops.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=(interpolates),
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty2(netD, real_data, fake_data, real_hidden, fake_hidden):
    alpha = ops.rand(opt.batch_size, 1)
    alpha1 = alpha.expand(real_data.size())
    if opt.cuda:
        alpha1 = alpha1.cuda()
    interpolates = alpha1 * real_data + ((1 - alpha1) * fake_data)
    
    alpha2 = alpha.expand(real_hidden.size())
    if opt.cuda:
        alpha2 = alpha2.cuda()
    interpolates_hidden = alpha2 * real_hidden + ((1 - alpha2) * fake_hidden)

    if opt.cuda:
        interpolates = interpolates.cuda()
        interpolates_hidden = interpolates_hidden.cuda()
    
    interpolates.requires_grad_(True)
    interpolates_hidden.requires_grad_(True)
    disc_interpolates = netD(interpolates, interpolates_hidden)

    ones = ops.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=(interpolates, interpolates_hidden),
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)#[0]
    gradients = ops.cat((gradients[0], gradients[1]), dim=1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# build model
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
netD2 = model.Discriminator_D2(opt)
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)
param_dict = load_checkpoint("your_path/ckpt_of_pretrain_netdec.ckpt")
load_param_into_net(netDec, param_dict)

# Init Tensor
input_res = ops.randn(opt.batch_size, opt.resSize)
input_att = ops.randn(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = ops.randn(opt.batch_size, opt.nz)
input_res_unpair = ops.randn(opt.batch_size, opt.resSize)
input_att_unpair = ops.randn(opt.batch_size, opt.attSize)

# optimizer
optimizer = nn.Adam(netE.trainable_params(), lr=opt.lr)
optimizerD = nn.Adam(netD.trainable_params(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD2 = nn.Adam(netD2.trainable_params(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = nn.Adam(netG.trainable_params(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = nn.Adam(netF.trainable_params(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = nn.Adam(netDec.trainable_params(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))


for epoch in range(0, opt.nepoch):
    for loop in range(0, opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.trainable_params(): #unfreeze discriminator
                p.requires_grad = True
                
            for p in netD2.trainable_params(): #unfreeze discriminator2
                p.requires_grad = True 

            for p in netDec.trainable_params(): #unfreeze deocder
                p.requires_grad = True                
                
            # Train D1, D2 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #LAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()  
                netD2.zero_grad()
                
                input_resv = (input_res)
                input_attv = (input_att)
                input_res_unpairv = (input_res_unpair)
                input_att_unpairv = (input_att_unpair)
                
                
                
                input_fake_att_unpairv = (netDec(input_res_unpairv)) 
                
                # Train the Decoder.
                
                def forward_fn_1(features, attributes):
                    recon = netDec(features)
                    loss = opt.recons_weight * WeightedL1(recons, input_attv) 
                    return loss
                grad_fn = mindspore.value_and_grad(forward_fn_1, None, optimizerDec.parameters, has_aux=True)
                loss, grads = grad_fn(input_resv, input_attv)
                optimizerDec(grads)
                
                def forward_fn_2():
                    errD = 0
                    # Train the Discriminator on real data.
                    criticD_real = netD(input_resv, input_attv)
                    criticD_real = opt.gammaD * criticD_real.mean()
                    errD += -criticD_real                               

                    criticD_real_v_unpair = netD2(input_res_unpairv, input_fake_att_unpairv)
                    criticD_real_v_unpair = opt.gammaD * criticD_real_v_unpair.mean()
                    errD += -criticD_real_v_unpair

                    if opt.encoded_noise:  
                        means, log_var = netE(input_resv, input_attv)
                        std = ops.exp(0.5 * log_var)
                        eps = ops.randn([opt.batch_size, opt.latent_size]).cpu()
                        if opt.cuda:
                            eps = Variable(eps.cuda())
                        else:
                            eps = Variable(eps)
                        z = eps * std + means 
                    else:
                        noise.normal_(0, 1)
                        z = Variable(noise)                    

                    # Train the Discriminator on fake data.
                    if loop == 1:
                        fake = netG(z, c=input_attv)
                        dec_out = netDec(fake)
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                    else:
                        fake = netG(z, c=input_attv)

                    criticD_fake = netD(fake.detach(), input_attv)
                    criticD_fake = opt.gammaD * criticD_fake.mean()
                    errD += criticD_fake

                    # on D2
                    noise.normal_(0, 1)
                    z = Variable(noise)

                    if loop == 1:
                        fake_unpair = netG(z, c=input_fake_att_unpairv)
                        dec_out = netDec(fake_unpair)
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake_unpair = netG(z, a1=opt.a1, c=input_fake_att_unpairv, feedback_layers=feedback_out)
                    else:
                        fake_unpair = netG(z, c=input_fake_att_unpairv)

                    criticD_fake_unpair = netD2(fake_unpair.detach(), input_fake_att_unpairv)
                    criticD_fake_unpair = opt.gammaD * criticD_fake_unpair.mean()
                    errD += criticD_fake_unpair

                    # gradient penalty
                    gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_attv)
                    errD += gradient_penalty
                    gp_sum += gradient_penalty.data                 

                    # gradient penalty for unpair.  
                    gradient_penalty_v_unpair = opt.gammaD * calc_gradient_penalty(netD2, input_res_unpair, fake_unpair.data, input_fake_att_unpairv)    
                    errD += gradient_penalty_v_unpair
                    gp_sum += gradient_penalty_v_unpair.data
                    return errD
                
                grad_fn = mindspore.value_and_grad(forward_fn_2, None, optimizerD.parameters, has_aux=True)
                grad_fn_2 = mindspore.value_and_grad(forward_fn_2, None, optimizerD2.parameters, has_aux=True)
                loss, grads = grad_fn()
                optimizerD(grads)
                loss, grads = grad_fn_2()
                optimizerD2(grads)
                
                # if opt.lambda_mult == 1.1:
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                
                # non-conditional D, Wasserstein distance
                Wasserstein_D_v2 = criticD_real_v_unpair - criticD_fake_unpair
                D_cost_v2 = criticD_fake_unpair - criticD_real_v_unpair + gradient_penalty_v_unpair

            gp_sum /= (2 * opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1


            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.trainable_params(): #freeze discrimator
                p.requires_grad = False
            for p in netD2.trainable_params(): #freeze discrimator
                p.requires_grad = False 
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.trainable_params(): #freeze decoder
                    p.requires_grad = False
            
            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()

            input_resv = (input_res)
            input_attv = (input_att)
            input_res_unpairv = (input_res_unpair)
            input_att_unpairv = (input_att_unpair)
            
            input_fake_att_unpairv = (netDec(input_res_unpairv))
    
            # seen class
            def forward_fn_3():
                pass
                means, log_var = netE(input_resv, input_attv)
                std = ops.exp(0.5 * log_var)
                eps = ops.randn([opt.batch_size, opt.latent_size]).cpu()
                if opt.cuda:
                    eps = Variable(eps.cuda())
                else:
                    eps = Variable(eps)
                z = eps * std + means

                if loop == 1:
                    recon_x = netG(z, c=input_attv)
                    dec_out = netDec(recon_x)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    recon_x = netG(z, c=input_attv)

                # The seen vae loss
                vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well             
                errG = vae_loss_seen 
            
                # net G fake data.
                if opt.encoded_noise:
                    criticG_fake = netD(recon_x, input_attv).mean()
                    fake = recon_x 
                else:
                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    if loop == 1:
                        fake = netG(noisev, c=input_attv)
                        dec_out = netDec(fake) #Feedback from Decoder encoded output
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                    else:
                        fake = netG(noisev, c=input_attv)
                    criticG_fake = netD(fake, input_attv).mean()
                
                # net G unpair fake data. (Do not need to regenerate.)
                noise.normal_(0, 1)
                z = Variable(noise)
                if loop == 1:
                    fake_unpair = netG(z, c=input_fake_att_unpairv)
                    dec_out = netDec(fake_unpair)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake_unpair = netG(z, a1=opt.a2, c=input_fake_att_unpairv, feedback_layers=feedback_out)
                else:
                    fake_unpair = netG(z, c=input_fake_att_unpairv)
                criticG_fake_v2 = netD2(fake_unpair, input_fake_att_unpairv).mean()           

                G_cost = -criticG_fake
                G_cost_v2 = -criticG_fake_v2
                errG += opt.gammaG * G_cost + opt.gammaG * G_cost_v2

                # seen
                netDec.zero_grad()
                recons_fake = netDec(fake)
                R_cost = WeightedL1(recons_fake, input_attv)
                errG += opt.recons_weight * R_cost

                # unseen (Need to regenerate.)
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake_unpair = netG(noisev, c=input_att_unpairv)
                    dec_out = netDec(fake_unpair) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake_unpair = netG(noisev, a1=opt.a1, c=input_att_unpairv, feedback_layers=feedback_out)
                else:
                    fake_unpair = netG(noisev, c=input_att_unpairv)

                fake_unpair = model.grad_scale(fake_unpair, scale=10.0)

                fake_unpair_att = netDec(fake_unpair)
                R_cost_unpair_att = WeightedL1(fake_unpair_att, input_att_unpairv)
                errG += opt.recons_weight * R_cost_unpair_att  
                return errG
            
            grad_fn = mindspore.value_and_grad(forward_fn_3, None, optimizer.parameters, has_aux=True)
            loss, grads = grad_fn()
            optimizer(grads)
            
            grad_fn = mindspore.value_and_grad(forward_fn_3, None, optimizerG.parameters, has_aux=True)
            loss, grads = grad_fn()
            optimizerG(grads)
            
            if loop == 1:
                grad_fn = mindspore.value_and_grad(forward_fn_3, None, optimizerF.parameters, has_aux=True)
                loss, grads = grad_fn()
                optimizerF(grads)
                
            if opt.recons_weight > 0 and not opt.freeze_dec:
                grad_fn = mindspore.value_and_grad(forward_fn_3, None, optimizerDec.parameters, has_aux=True)
                loss, grads = grad_fn()
                optimizerDec(grads)





















