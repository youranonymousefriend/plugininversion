from model import model_library


def main():
    names = [m.name for m in model_library]
    for i in range(len(names)):
        print(f'{i} : {names[i]}')
    print(names)


if __name__ == '__main__':
    main()
