import struct

def read_file(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    split_elements = data.split(b'\r\n')

    ret = [[], [], []]

    for count, element in enumerate(split_elements):
        element = element.rstrip(b'\r\n')

        if count == 0:
            ret[count] = list(struct.unpack(f'<{len(element)}c', element))
        elif count == 1:
            ret[count] = list(struct.unpack(f'<{len(element) // 4}I', element))
        elif count == 2:
            ret[count] = list(struct.unpack(f'<{len(element) // 4}f', element))
    return ret


def write_file(file, datas, type):
    for data in datas:
        file.write(struct.pack(type, data))
    file.write(b'\r\n')

if __name__ == '__main__':
    filepath = 'mybin3.bin'
    datas = read_file(filepath)

    print(datas)

    for data in datas:
        data.sort()

    print(datas)

    data1, data2, data3 = datas

    f = open(filepath, 'wb+')
    write_file(f, data1, 'c')
    write_file(f, data2, 'I')
    write_file(f, data3, 'f')
    f.close()

    datas = read_file(filepath)
    print(datas)
