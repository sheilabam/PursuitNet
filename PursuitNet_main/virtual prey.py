import math
import pandas as pd


def calculate_distance(prey_pos, predator_pos):
    x = prey_pos[0] - predator_pos[0]
    y = prey_pos[1] - predator_pos[1]
    # print("nihao:", prey_pos[0],prey_pos[1], predator_pos[0], predator_pos[1])
    distance = math.sqrt(x ** 2 + y ** 2)
    return round(distance, 2)


def calculate_parameters(borx, bory, micex, micey):
    dis_prez = 46
    dis_safez = 122
    v = 15
    zhongxz = 325
    zhongyz = 241
    shangz = 22
    xiaz = 461
    zuoz = 106
    youz = 544
    r = 180
    R = 221
    rfangz = r ** 2
    Rfangz = R ** 2

    xianxx = borx - zhongxz
    xianxz = xianxx
    xianyz = bory - zhongyz

    pandingxz = (borx - zhongxz) ** 2
    pandingyz = (bory - zhongyz) ** 2

    chayy = bory - micey
    chay = chayy if chayy != 0 else 1
    bizhichu = abs((micex - borx) / chay)
    bizhi = bizhichu
    bizhifang = bizhi ** 2
    speedx = math.sqrt(v ** 2 / (bizhifang + 1))
    speedy =  bizhi * speedx

    # print(f"Prey speedx: {speedx}, speedy: {speedy}")
    return speedx, speedy, dis_prez, dis_safez, zhongxz, zhongyz, shangz, xiaz, zuoz, youz, xianxz, xianyz, rfangz, Rfangz, pandingxz, pandingyz


def update_prey_position(predator_pos, prey_pos):
    reald = calculate_distance(predator_pos, prey_pos)
    speedx, speedy, dis_pre, dis_safe, zhongx, zhongy, shang, xia, zuo, you, xianx, xiany, rfang, Rfang, pandingx, pandingy = calculate_parameters(predator_pos[0], predator_pos[1], prey_pos[0], prey_pos[1])
    print(f"Real distance between predator and prey: {reald}")

    if reald <= dis_pre:
        print('predation!!!')
        predator_pos[0] = predator_pos[0]
        predator_pos[1] = predator_pos[1]
        return prey_pos

    elif dis_pre < reald <= dis_safe:
        if pandingx + pandingy <= rfang:

            if predator_pos[0] >= prey_pos[0] and predator_pos[1] <= prey_pos[1]:
                print('1')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif predator_pos[0] < prey_pos[0] and predator_pos[1] < prey_pos[1]:
                print('2')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif predator_pos[0] <= prey_pos[0] and predator_pos[1] >= prey_pos[1]:
                print('3')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif predator_pos[0] > prey_pos[0] and predator_pos[1] > prey_pos[1]:
                print('4')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
        elif pandingx + pandingy > rfang and pandingx + pandingy < Rfang and zhongx <= prey_pos[0]< you and shang < prey_pos[1] < zhongy:

            if ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy < predator_pos[1] and predator_pos[1] >= prey_pos[1]:
                print('21')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy <= predator_pos[1] and predator_pos[1] < prey_pos[1]:
                print('22')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy >= predator_pos[1] and predator_pos[0] > prey_pos[0]:
                print('23')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy > predator_pos[1] and predator_pos[0] <= prey_pos[0]:
                print('24')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
        elif pandingx + pandingy > rfang and pandingx + pandingy < Rfang and zuo < prey_pos[0] < zhongx and shang < prey_pos[1] <= zhongy:

            if ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy < predator_pos[1] and predator_pos[1] >= prey_pos[1]:
                print('31')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy <= predator_pos[1] and predator_pos[1] < prey_pos[1]:
                print('32')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy >= predator_pos[1] and predator_pos[0] < prey_pos[0]:
                print('33')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy > predator_pos[1] and predator_pos[0] >= prey_pos[0]:
                print('34')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
        elif pandingx + pandingy > rfang and pandingx + pandingy < Rfang and zuo < prey_pos[0] <= zhongx and zhongy < prey_pos[1] < xia:

            if ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy < predator_pos[1] and predator_pos[0] >= prey_pos[0]:
                print('41')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy <= predator_pos[1] and predator_pos[0] > prey_pos[0]:
                print('42')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy >= predator_pos[1] and predator_pos[1] > prey_pos[1]:
                print('43')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy > predator_pos[1] and predator_pos[1] <= prey_pos[1]:
                print('44')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
        elif pandingx + pandingy > rfang and pandingx + pandingy < Rfang and zhongx < prey_pos[0] < you and zhongy <= prey_pos[1] < xia:

            if ((predator_pos[0] - zhongx) / xianx) * xiany + zhongy > predator_pos[1] and predator_pos[0] >= prey_pos[0]:
                print('51')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((prey_pos[0] - zhongx) / xianx) * xiany + zhongy >= predator_pos[1] and predator_pos[0] < prey_pos[0]:
                print('52')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] -= speedy * 0.033 * 5.73
            elif ((prey_pos[0] - zhongx) / xianx) * xiany + zhongy <= predator_pos[1] and predator_pos[1] < prey_pos[1]:
                print('53')
                predator_pos[0] -= speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
            elif ((prey_pos[0] - zhongx) / xianx) * xiany + zhongy < predator_pos[1] and predator_pos[1] >= prey_pos[1]:
                print('54')
                predator_pos[0] += speedx * 0.033 * 5.73
                predator_pos[1] += speedy * 0.033 * 5.73
        return predator_pos

    elif reald > dis_safe:
        print('safe!!!')
        predator_pos[0] = predator_pos[0]
        predator_pos[1] = predator_pos[1]
        return predator_pos

df = pd.read_excel('C:/Users/dell/Desktop/demo1/i.xlsx')


prey_pos = [df.iloc[0]['x2'], df.iloc[0]['y2']]
predator_pos = [df.iloc[0]['x1'], df.iloc[0]['y1']]

results = []

for index, row in df.iterrows():
    if index == 0:
        results.append(prey_pos.copy())
        continue
    predator_pos = [row['x1'], row['y1']]
    prey_pos = update_prey_position(prey_pos, predator_pos)
    results.append(prey_pos.copy())

df['x2'] = [round(pos[0]) for pos in results]
df['y2'] = [round(pos[1]) for pos in results]


output_file = 'C:/Users/dell/Desktop/demo1/output.xlsx'
df.to_excel(output_file, index=False)
