from model import model_library
from datasets import image_net, weird_image_net
from utils import exp_starter_pack


def main():
    exp_name, args, _ = exp_starter_pack()

    network = args.network
    model, image_size, batch_size, name = model_library[network]()
    score = 0
    data = image_net.eval() if image_size == 224 else weird_image_net.eval()

    for i in range(10):
        x, y = data[-i - 1]
        x = x.cuda()
        score += (model(x.unsqueeze(0)).argmax() == y)
    with open('fake_acc.txt', 'a') as f:
        print(f'{name}\t{score.item()}', file=f, flush=True)


if __name__ == '__main__':
    main()
