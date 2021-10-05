from model import model_library
import pdb

chosen = [0, 1, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 32, 33, 34, 35, 36, 40,
          41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 59, 62, 63, 64, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76,
          79, 85, 87, 88, 89, 90, 91, 92, ]


def main():
    names = [m.name for m in model_library]
    print(len(names))
    count = 0
    for i in chosen:
        print(f'{names[i]}', end='')
        count += 1
        if count == 8:
            count = 0
            print(' \\\\')
        else:
            print(' & ', end='')
    for i in chosen:
        cur_name = names[i]
        family = cur_name.split('_', 1)[0]
        cur_name = cur_name.split('_', 1)[1]
        if cur_name[0] in '0987654321':
            cur_name = cur_name.split('_', 1)[1]
        alias = cur_name
        alias = alias.replace('distilled', 'D')
        alias = alias.replace('patch', 'p')
        alias = alias.replace('base', '')
        alias = alias.replace('window', 'w')
        out = '\\text{' + family + '} & \\text{' + alias + '} & \\text{' + cur_name + '} & TorchVision & \\cite{Bengio+chapter2007} \\\\'
        out = out.replace('_', '\\_')
        print(out)
    for i in chosen:
        out = '\\newcommand{\\model' + str(i) + '}{}'
        print(out)


if __name__ == '__main__':
    main()
