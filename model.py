import mindspore.nn as nn
from mindspore.common.initializer import Normal

#Encoder
class Encoder(nn.Cell):

    def __init__(self, opt):

        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size 
        self.fc1 = nn.Dense(layer_sizes[0], layer_sizes[-1])
        self.fc3 = nn.Dense(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.linear_means = nn.Dense(latent_size*2, latent_size)
        self.linear_log_var = nn.Dense(latent_size*2, latent_size)

    def construct(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

#Decoder/Generator
class Generator(nn.Cell):

    def __init__(self, opt):

        super(Generator,self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        input_size = latent_size * 2 
        self.fc1 = nn.Dense(input_size, layer_sizes[0])
        self.fc3 = nn.Dense(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def construct(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z, c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x

# conditional discriminator for inductive
class Discriminator_D1(nn.Cell):
    def __init__(self, opt): 
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Dense(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Dense(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h 

class AttDec(nn.Cell):
    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Dense(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Dense(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2)
        self.hidden = None
        self.sigmoid = None

    def construct(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None: 
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()
    
    def getHiddenOut(self):
        return self.hidden

#Feedback Modules
class Feedback(nn.Cell):
    def __init__(self,opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Dense(opt.ngh, opt.ngh)
        self.fc2 = nn.Dense(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self,x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h

class Discriminator_D2(nn.Cell):
    def __init__(self, opt): 
        super(Discriminator_D2, self).__init__()
        self.fc1 = nn.Dense(opt.resSize + 4096, opt.ndh)
        self.fc2 = nn.Dense(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, h):
        x = torch.cat((x, h), dim=1)
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h 






















































