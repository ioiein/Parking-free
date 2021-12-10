from math import sin, cos

import cv2
import numpy as np

"""
Контуры дорожного полотна зависят от параметров сетки (как плотно или широко будут отрисовываться автомобили).
И параметры сетки И контуры дорожного полотна зависят от размеров карты.
Все эти константы-параметры были подбраны вручную.
"""
# размеры карты
MAP_WIDTH = 1024
MAP_HEIGHT = 256

# параметры сетки
GRID_WIDTH = 10
X_AR = 500
D_AR = - GRID_WIDTH * 0.008

# контур дорожного полотна
ROAD_POINTS = [
    (0, 145), (340, 145), (340, 100), (555, 100), (585, 10), (920, 10),   # сверху
    (1015, 80), (1015, 256),  # справа
    (940, 256), (940, 245), (0, 245), (0, 145)  # снизу
]

# ширина машины
CAR_WIDTH = 16

# ручная корректировка
CORRECTION_AREA = [[985, 120], [985, 200], [1020, 200], [1020, 120], [985, 120]]


# матрица проективного преобразования (позволяет сменить ракурс)
def homography_matrix(map_width: int = MAP_WIDTH, map_height: int = MAP_HEIGHT):

    source_points = np.asarray([
      [ 0, 220],  # top left
      [ 1290, 400], # top mid
      [ 1890, 640], # top right
      [ 1920, 915], # bot right
      [ 1500, 1080], # bot mid
      [   0,  840]  # bot left
    ])

    target_points = np.asarray([
      [0, 0],
      [map_width // 2, 0],
      [map_width, 0],
      [map_width, map_height],
      [map_width // 2, map_height],
      [0, map_height]
    ])

    return cv2.findHomography(source_points, target_points)


def make_grid(
        grid_width: int = GRID_WIDTH,
        map_width: int = 1010, # MAP_WIDTH
        x_ar: int = X_AR,
        d_ar: float = D_AR,
):
    """
    Cетка для спроецированного изображения (для каждого bbox будет определяться - к какой линии он ближе)
    Cлева и справа частота сетки увеличиватся - за счет этого удается "раздвинуть" машины на итоговой схеме.
    Параметры подбираются вручную
    :param grid_width: начальная частота сетки
    :param map_width: ширина карты-схемы
    :param x_ar: кордината х, от которой влева и вправо сетка учащается (по арифметической прогресссии)
    :param d_ar: разность арифметической прогрессии
    :return: координаты х сетки
    """
    grids_source_lst = []

    # сетка арфметическая влево
    grid_width_left = grid_width
    x = x_ar
    while x > 0:
        assert grid_width_left > 0, 'please, reduce abs(d_ar) to make smaller steps'
        grids_source_lst.append(int(x))
        grid_width_left += d_ar  # элемент а.п.
        x -= grid_width_left  # смещаюсь влево
    grids_source_lst = grids_source_lst[::-1] # разворачиваю список

    # сетка арфметическая вправо
    grid_width_right = grid_width
    x = x_ar
    while x < map_width:
        assert grid_width_right > 0, 'please, reduce abs(d_ar) to make smaller steps'
        grids_source_lst.append(int(x))
        grid_width_right += d_ar  # элемент а.п.
        x += grid_width_right  # смещаюсь вправо

    grids_source_lst.remove(x_ar)  # убираю стартовый х, учтенный дважды (при движении влево и при движении вправо)

    return grids_source_lst


def make_map(n_lines: int, map_width: int = MAP_WIDTH, map_height: int = MAP_HEIGHT, road_points: list = ROAD_POINTS):
    """
    Создаю карту-схема.
    Отрисовываю дорожное полотоно.
    "Переношу" сетку с проекции на карту-схему, но уже равномерно!
    За счет равномерности области, которые имели плотную сетку, "разряжаются", а области с редкой сеткой - "сужаются".
    Благодаря этому, карта-схема выравнивается по сравнению с проекционным изображением.
    :param map_width: ширина карты
    :param map_height: высота карты
    :param n_lines: количество линий в сетке
    :return:
    """
    # шаблон карты
    pk_map = np.full((map_height, map_width, 3), 150, dtype='uint8')

    # отрисовка дорожного полотнаи
    for i in range(len(road_points) - 1):
        cv2.line(pk_map, tuple(road_points[i]), tuple(road_points[i + 1]), color=(255, 255, 255), thickness=4)
    pk_map = cv2.fillPoly(pk_map, pts=[np.array(road_points)], color=(255, 255, 255))

    # определяю координаты х равномерной сетки
    grid_width = map_width / (n_lines + 1)  # линии делят карту на n_lines + 1 область
    grids_target_lst = []
    for i in range(1, n_lines + 1):
        x = int(grid_width * i)
        grids_target_lst.append(x)

    return pk_map, grids_target_lst


# используется для нахождения ближайшей линии к центру bbox
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def draw_car(
        pk_map: np.array,
        x: int,
        y: int,
        pos: int,
        y_level: int,
        grids_s_lst: list,
        grids_t_lst: list,
        car_width: int = CAR_WIDTH):
    """
    Отрисовка 1 машины на карте-схеме.
    :param pk_map: карта-схема,
    :param x: центр авто,
    :param y: центр авто,
    :param pos: угол относительно оси ординат, под которым рисуется длинная сторона машины (0 - вертикально, 90 - горизонтально, 125 - под углом)
    :param y_level: координата по оси ординат, на которой будет отрисовываться авто (х - рассчитывается, y задается через y_level)
    :param grids_s_lst: сетка спроецированной фотки
    :param grids_t_lst: сетка карты-схемы
    :param car_width: ширина машины
    :return: карта-схема
    """
    color = (150, 150, 150)
    color = (0, 0, 255)
    car_length = 2.3 * car_width

    # TODO возможно без этой встравки?
    # если в трансформированном фото точка попадает в указанную область, то я ее немного смещаю
    if correction_right_bottom_parking(x, y):
        x = 1001

    idx = find_nearest(grids_s_lst, x)
    x = grids_t_lst[idx]  # смещаю на ближаюшую сетку
    if y_level is not None:
        y = y_level

    # рисую прямоугольник сверху вниз
    if pos == 0:
        pt1 = (int(x - car_width / 2), int(y - car_length / 2))
        pt2 = (int(x + car_width / 2), int(y + car_length / 2))
        pk_map = cv2.rectangle(pk_map, pt1, pt2, color, -1)

    # рисую прямоугольник справа налево
    elif pos == 90:
        pt1 = (int(x - car_length / 2), int(y - car_width / 2))
        pt2 = (int(x + car_length / 2), int(y + car_width / 2))
        pk_map = cv2.rectangle(pk_map, pt1, pt2, color, -1)

    # рисую под углом
    else:
        a = car_width

        Ax = int(x - car_length / 2 * sin(pos) - car_width / 2 * cos(pos))
        Dx = int(x - car_length / 2 * sin(pos) + car_width / 2 * cos(pos))

        Ay = int(y - car_length / 2 * cos(pos) + car_width / 2 * sin(pos))
        Dy = int(y - car_length / 2 * cos(pos) - car_width / 2 * sin(pos))

        Bx = int(x + car_length / 2 * sin(pos) - car_width / 2 * cos(pos))
        Cx = int(x + car_length / 2 * sin(pos) + car_width / 2 * cos(pos))

        By = int(y + car_length / 2 * cos(pos) + car_width / 2 * sin(pos))
        Cy = int(y + car_length / 2 * cos(pos) - car_width / 2 * sin(pos))

        rectangle_with_slope = np.array([[Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy]])
        cv2.fillPoly(pk_map, pts=[rectangle_with_slope], color=color)

    return pk_map


def make_contour(points: list, map_width: int = MAP_WIDTH, map_height: int = MAP_HEIGHT):
    """
    Функция для создания объекта Contour.
    По сути - это просто граница определенной праковочной области.
    Этот объект далее используется для проверки, какие bbox к нему относятся.
    :param map_width:
    :param map_height:
    :param points:
    :return:
    """

    # создаю пустую область
    src = np.zeros((map_height, map_width), dtype=np.uint8)

    # наношу линии
    for i in range(len(points) - 1):
        cv2.line(src, tuple(points[i]), tuple(points[i + 1]), color=(255, 0, 0), thickness=3)

    # получаю объект contour
    contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def draw_parking(
        bbox_lst: list,
        parking_map: np.array,
        contours,
        grids_source_lst: list,
        grids_target_lst: list,
        y_level: int,
        pos: int,
        capacity: int = 0
):
    """
    Отрисовка автомобилей на карте, которые находятся в пределах области "contours"
    :param bbox_lst: набор bbox
    :param parking_map: карта-схема
    :param contours: контур
    :param grids_source_lst: сетка спроецированного изображения
    :param grids_target_lst: сетка карты-схемы
    :param y_level: у координата отрисовки (х - рассчитывается автоматически)
    :param pos: угол отрисовки
    :param capacity: емкость парковки
    :return: карта-схема
    """

    # наношу размету
    for bbox in bbox_lst:
        if cv2.pointPolygonTest(contours[0], (bbox[0], bbox[1]), False) == 1:
            capacity -= 1
            parking_map = draw_car(
                pk_map=parking_map,
                x=bbox[0],
                y=bbox[1],
                pos=pos,
                y_level=y_level,
                grids_s_lst=grids_source_lst,
                grids_t_lst=grids_target_lst
            )

    # ограничение на случай, если машины запаркованы плотнее стандартного состояния
    if capacity < 0:
        capacity = 0

    return parking_map, capacity


def correction_right_bottom_parking(x: int, y: int, area: list = CORRECTION_AREA):
    # корректировка правой нижней области, которая возникает из-за изгиба изображения
    contour = make_contour(area)
    return cv2.pointPolygonTest(contour[0], (x, y), False) == 1