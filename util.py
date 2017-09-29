
def print_params(net):
    print('\nArchitecture: ')
    print(net)
    print('\nParameters:')
    params = list(net.parameters())
    for i in range(len(params)):
        print i, params[i].size()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)