import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class ClassificationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_head(self.classification_heads)

    def forward(self, x):
        """Sequentially pass `x` trough an encoder model and heads"""
        features = self.encoder(x)
        features = self.inter_activations + features[-1:]
        if self.layer_ensembles:
            all_labels = [classification_head(output) for classification_head, output in zip(self.classification_heads, self.inter_activations)]
        # Skip the first block, which is the stem of the encoder
        # features = features[1:]
        # if self.layer_ensembles:
        #     all_labels = [classification_head(output) for classification_head, output in zip(self.classification_heads, features)]
        else:
            all_labels = self.classification_heads(features[-1])
        return all_labels

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
