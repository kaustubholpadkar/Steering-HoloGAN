from hologan import util


data_dir = "../images/"
bs = 4

dataloader, n = util.get_data_loader(data_dir, bs)
print(n, "images")

i = 0
for batch, _ in dataloader:
    print(batch.shape)
    i += 1
