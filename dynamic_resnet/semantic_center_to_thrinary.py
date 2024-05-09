import torch

def gen_noise(weight, noise):
    new_w = weight * noise * torch.randn_like(weight)
    return new_w.to(weight.device)

semantic_center = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\research\\weight\\Resnet_raw_2.pth')
# print(semantic_center)

semantic_center_ternary = []
for item in semantic_center:
    semantic = item[0]
    label = item[1]
    semantic_layer_ternary = []
    for semantic_layer in semantic:
        print(semantic_layer)
        ctx_max, ctx_min = torch.max(semantic_layer), torch.min(semantic_layer)
        # lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        # higher_interval = ctx_max - (ctx_max - ctx_min) / 3
        lower_interval = 0.5
        higher_interval = 1.5
        out = torch.where(semantic_layer < lower_interval,
                          torch.tensor(0.).to(semantic_layer.device, semantic_layer.dtype), semantic_layer)
        out = torch.where(semantic_layer > higher_interval,
                          torch.tensor(2.).to(semantic_layer.device, semantic_layer.dtype), out)
        out = torch.where((semantic_layer >= lower_interval) & (semantic_layer <= higher_interval),
                          torch.tensor(1.).to(semantic_layer.device, semantic_layer.dtype), out)
        # out = out + gen_noise(out, 0.10)
        print(out)
        semantic_layer_ternary.append(out)

    semantic_center_ternary.append((semantic_layer_ternary, label))

torch.save(semantic_center_ternary, 'ResNet_semantic_center_mnist_ternary_hardware.pth')
eee=torch.load('ResNet_semantic_center_mnist_ternary_hardware.pth')

print(eee)
print('save sc')
# print(torch.load('Resnet_1116_noised.pth'))