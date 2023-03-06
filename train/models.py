import torch


class DomainPredictor(torch.nn.Module):
    def __init__(self, n_centers, fc_input_features, cnn_to_use):
        super(DomainPredictor, self).__init__()
        # domain predictor
        self.fc_feat_in = fc_input_features
        self.n_centers = n_centers

        if 'resnet18' in cnn_to_use:
            self.E = 128

        elif 'resnet34' in cnn_to_use:
            self.E = 128

        elif 'resnet50' in cnn_to_use:
            self.E = 256

        elif 'densenet121' in cnn_to_use:
            self.E = 128

        self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.domain_classifier = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)

        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)
        if self.n_centers > 1:
            self.activation = torch.nn.Softmax(dim=1)
        else:
            self.activation = torch.nn.Sigmoid()

    def forward(self, x):

        dropout = torch.nn.Dropout(p=0.1)
        domain_emb = self.domain_embedding(x)
        domain_emb = dropout(domain_emb)
        domain_prob = self.activation(self.domain_classifier(domain_emb))
        return domain_prob


class Encoder(torch.nn.Module):
    def __init__(self, dim, cnn_to_use, embedding_bool=True, n_domains=5):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Encoder, self).__init__()

        self.used_cnn = cnn_to_use
        self.embedding_bool = embedding_bool

        pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', self.used_cnn, pretrained=True)

        if ('resnet' in self.used_cnn) or ('resnext' in self.used_cnn):
            fc_input_features = pre_trained_network.fc.in_features
        elif 'densenet' in self.used_cnn:
            fc_input_features = pre_trained_network.classifier.in_features
        elif 'mobilenet' in self.used_cnn:
            fc_input_features = pre_trained_network.classifier[1].in_features

        self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

        if torch.cuda.device_count() > 1:
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)

        self.fc_feat_in = fc_input_features

        self.dim = dim

        if self.embedding_bool:

            if 'resnet34' in self.used_cnn:
                self.E = self.dim
                self.L = self.E
                self.D = 64

            elif 'resnet50' in self.used_cnn:
                self.E = self.dim
                self.L = self.E
                self.D = 128

            elif 'resnet152' in self.used_cnn:
                self.E = self.dim
                self.L = self.E
                self.D = 128

            self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)

        self.domain_predictor = DomainPredictor(n_domains, self.E, self.used_cnn)
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, mode, alpha):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        dropout = torch.nn.Dropout(p=0.2)

        if x is not None:
            conv_layers_out = self.conv_layers(x)
            conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

        if 'mobilenet' in self.used_cnn:
            conv_layers_out = dropout(conv_layers_out)

        if self.embedding_bool:
            embedding_layer = self.embedding(conv_layers_out)
            embedding_layer = self.prelu(embedding_layer)
            features_to_return = embedding_layer
            normalized_output = torch.nn.functional.normalize(features_to_return, dim=1)

            if mode == "train":
                reverse_feature = ReverseLayerF.apply(embedding_layer, alpha)
                output_domain = self.domain_predictor(reverse_feature)
                return normalized_output, output_domain

            return normalized_output

        else:
            features_to_return = conv_layers_out
            normalized_output = torch.nn.functional.normalize(features_to_return, dim=1)

            if mode == "train":
                reverse_feature = ReverseLayerF.apply(features_to_return, alpha)
                output_domain = self.domain_predictor(reverse_feature)
                return normalized_output, output_domain

            return normalized_output


# reverse autograd
class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
