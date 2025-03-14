import torch.nn as nn
import math
import numpy as np
import torch 


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x     
        
class SCANet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SCANet, self).__init__()
        self.conv = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.cbam  = CBAM(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=[4])
        self.fc = nn.Linear(4608, num_classes) 
        self.s = nn.Sigmoid()   
        self.l = nn.ReLU6()
        self.rl = nn.ReLU()
        
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * 1 * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # train and val
    def forward(self, target, all, all_label, target_label ):
        """
        Spectra Metrics Learning.
        """
        # produce features
        encoded_images = []
        pred_results = []
        for i in np.arange(all.size(0)):

            x = all[i,:,:].unsqueeze(0)

            x = self.conv(x)
            x = self.bn1(x)
            x = self.relu(x)   
            # x = self.cbam(x)  #
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)        
            x = x.view(x.size(0), -1)
            feature = x
            encoded_images.append(feature)          

        # produce embeddings for target
        for i in np.arange(target.size(0)):

            x=target[i,:,:].unsqueeze(0) 
            x = self.conv(x)            
            x = self.bn1(x)
            x = self.relu(x)   
            # x = self.cbam(x)  
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)        
            x = x.view(x.size(0), -1)
            
            encoded_images.append(x)
            outputs = torch.stack(encoded_images)  


            # get similarity between features and target
            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = similarities.t()
            # print(similarities)

            # produce predictions for target probabilities
            preds = self.classify(similarities,support_set_y=all_label)
            # print(preds)
            fx = self.fc(x)
            fx = self.s(fx)
            fpreds = (fx + preds)/2
            # print(fpreds)
            pred_results.append(fpreds)
            pred = torch.stack(pred_results) 
            encoded_images.pop()
            
        return pred

    def predict(self, target, all_feature, all_label):
        """
        Spectra Metrics Learning for prediction.
        """
        # get features
        encoded_images = []
        pred_results = []
        encoded_images.append(all_feature)

        # produce embeddings for target 
        for i in np.arange(target.size(0)):
            x=target[i,:,:].unsqueeze(0) 
            x = self.conv(x)            
            x = self.bn1(x)
            x = self.relu(x)  
            # x = self.cbam(x)  
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)        
            x = x.view(x.size(0), -1)
            
            x = x.unsqueeze(1)
            
            outputs = torch.cat([all_feature,x], dim=0)


            # get similarity between features and target
            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = similarities.t()

            # produce predictions for target probabilities
            preds = self.classify(similarities,support_set_y=all_label)
            # print(preds)
            fx = self.fc(x)
            fx = self.s(fx)
            # print(fx)
            fpreds = (fx + preds)/2
            print(fpreds)
            pred_results.append(fpreds)
            pred = torch.stack(pred_results) 
            
        return pred       


class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        get similarities
        """
        eps = 1e-10
        similarities = []
        sum_input = torch.sum(torch.pow(input_image, 2))
        input_magnitude = sum_input.clamp(eps, float("inf")).rsqrt()
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2))
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.mm(support_image.squeeze().unsqueeze(1))
            cosine_similarity = dot_product * support_magnitude *input_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        similarities = similarities.squeeze(1) 
        return similarities

        
class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        get similarity loss
        """
        softmax = nn.Softmax()
        sigmoid = nn.Sigmoid()
        softmax_similarities = softmax(similarities)  
        preds = softmax_similarities.mm(support_set_y)

        return preds        