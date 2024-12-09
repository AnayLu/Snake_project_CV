# импорт необходимых библиотек
import cv2
import numpy as np
import mediapipe as mp
import time
import math

# определяем переменные, некоторые из них константы
score = 0  # отвечает за счет, изменяется
max_score = 11  # константа, сколько очков необходимо набрать, чтобы победить
list_capacity = 0
max_lc = 20
crit_dist = 35
l = []
flag = 0  # отвечает за режим игры, изменяется
apple_radius = 15  # константа, радиус яблока
apple_x, apple_y, center = None, None, None  # координаты центра яблока
snake = []  # список, хранящий в себе сегменты змейки
scr = 0
x_tip, y_tip = None, None  # координаты кончика указательного пальца
crit_time = 2  # критическое время, режимы ниндзя и быстрая змейка, зависит от выбранной сложности
start_time = time.time()  # начальное время
max_time = 30  # время сбора очков для жадной змейки
maxscore = 0  # максимальный результат, для жадной змейки
status = 'normal'  # статус для разных концовок режимов
snake2 = []  # список, в котором перечислены отрезки змейки
mode = 0


class Point:  # класс точки
    def __init__(self, x, y=None, polar=False):
        if isinstance(x, Point):
            y = x.y
            x = x.x
        self.x = x
        self.y = y
        self.polar = polar
        if polar:
            self.y = math.sin(y) * x
            self.x = math.cos(y) * x

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def dist(self, x=None, y=None):
        if x is None:
            return self.__abs__()
        elif isinstance(x, Point):
            y = x.y
            x = x.x
        return Point(x - self.x, y - self.y).__abs__()

    def __str__(self):
        return f"({self.x}, {self.y})"


class Vector(Point):  # класс вектора
    def __init__(self, a, b=None, c=None, d=None):
        if isinstance(a, Point):
            if isinstance(b, Point):
                a, b = b.x - a.x, b.y - a.y
            else:
                a, b = a.x, a.y
        else:
            if isinstance(c, int):
                a, b = c - a, d - b
        super().__init__(a, b)

    def dot_product(self, other):
        return self.x * other.x + self.y * other.y

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot_product(other)
        else:
            return Vector(self.x * other, self.y * other)

    def cross_product(self, other):
        return self.x * other.y - self.y * other.x

    def __xor__(self, other):
        return self.cross_product(other)

    def __rmul__(self, other):
        return self * other

    def angle(self, other):
        return math.acos((self * other) / abs(self) / abs(other))

    def min_dist_segment(self, other):
        return abs(self ^ other) / abs(self)

    def dist_segment(self, p1, p2, p3):
        a = Vector(p2, p1)
        b = Vector(p3, p1)
        if self * a <= 0:
            return p1.dist(p2)
        elif self * b >= 0:
            return p1.dist(p3)
        else:
            return self.min_dist_segment(a)

    def dist_ray(self, p1, p2):
        a = Vector(p2, p1)
        if self * a <= 0:
            return p1.dist(p2)
        else:
            return self.min_dist_segment(a)


def snake_crossing(v1, v2):  # функция для определения пересечения отрезков, код взят у Дарьи Порай
    if v1[0] > v1[2]:
        v1 = [v1[2], v1[3], v1[0], v1[1]]
    A = Point(v1[0], v1[1])
    B = Point(v1[2], v1[3])
    if (Vector(A, B) ^ Vector(A, Point(v2[2], v2[3]))) * (Vector(A, B) ^ Vector(A, Point(v2[0], v2[1]))) > 0:
        return "NO"
    elif (Vector(A, B) ^ Vector(A, Point(v2[2], v2[3]))) * (Vector(A, B) ^ Vector(A, Point(v2[0], v2[1]))) == 0:
        fl = False
        if (Vector(A, B) ^ Vector(A, Point(v2[2], v2[3]))) == 0 and A.x <= v2[2] <= B.x and \
                min(A.y, B.y) <= v2[3] <= max(A.y, B.y):
            fl = True
        if (Vector(A, B) ^ Vector(A, Point(v2[0], v2[1]))) == 0 and A.x <= v2[0] <= B.x and \
                min(A.y, B.y) <= v2[1] <= max(A.y, B.y):
            fl = True
        if (Vector(A, B) ^ Vector(A, Point(v2[2], v2[3]))) == 0 and (
                Vector(A, B) ^ Vector(A, Point(v2[0], v2[1]))) == 0:
            if min(v2[0], v2[2]) <= A.x <= max(v2[0], v2[2]) and min(v2[1], v2[3]) <= A.y <= max(v2[1], v2[3]):
                fl = True
        if fl:
            return "YES"
        else:
            return "NO"
    else:
        if Vector(A, B) ^ Vector(A, Point(v2[2], v2[3])) > 0:
            v2 = [v2[2], v2[3], v2[0], v2[1]]
        C = Point(v2[0], v2[1])
        D = Point(v2[2], v2[3])
        if Vector(A, D) ^ Vector(A, C) < 0 or Vector(B, C) ^ Vector(B, D) < 0:
            return "NO"
        else:
            return "YES"


def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)  # считает расстояние между двумя точками


def choose_mode(cap, handsDetector):  # функция выбора сложности в ниндзя и быстрой змейке
    global mode
    global crit_time
    ret, frame = cap.read()  # берем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    cv2.rectangle(flippedRGB, (10, 750), (610, 900), (154, 214, 143), -1)  # рисуем прямоугольники-кнопки
    cv2.rectangle(flippedRGB, (1310, 750), (1910, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (660, 750), (1260, 900), (154, 214, 143), -1)
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)  # рисуем кончик указательного пальца
        if (x_tip is not None) and (y_tip is not None) and (10 <= x_tip <= 610) and (
                750 <= y_tip <= 900):  # условия на проверку куда мы "нажали"
            mode = 1
            crit_time = 6
        elif (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (750 <= y_tip <= 900):
            mode = 2
            crit_time = 4
        elif (x_tip is not None) and (y_tip is not None) and (1310 <= x_tip <= 1910) and (750 <= y_tip <= 900):
            mode = 3
            crit_time = 2
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # все отображаем на картинке
    cv2.putText(res_image, 'Easy', (230, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)  # печатаем текст на кнопках
    cv2.putText(res_image, 'Medium', (830, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Hard', (1530, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.imshow('Game', res_image)  # показываем что получилось
    if cv2.waitKey(1) == 65:
        return


def menu(cap, handsDetector):  # функция меню
    global flag
    ret, frame = cap.read()  # берем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    cv2.rectangle(flippedRGB, (10, 750), (610, 900), (154, 214, 143), -1)  # рисуем прямоугольники-кнопки
    cv2.rectangle(flippedRGB, (660, 50), (1260, 200), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (1310, 750), (1910, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (10, 350), (610, 500), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (1310, 350), (1910, 500), (154, 214, 143), -1)
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)  # рисуем кончик указательного пальца

        if (x_tip is not None) and (y_tip is not None) and (10 <= x_tip <= 610) and (
                750 <= y_tip <= 900):  # проверяем на какую кнопку мы "нажали"
            flag = 2
        elif (x_tip is not None) and (y_tip is not None) and (10 <= x_tip <= 610) and (350 <= y_tip <= 500):
            flag = 1
        elif (x_tip is not None) and (y_tip is not None) and (1310 <= x_tip <= 1910) and (350 <= y_tip <= 500):
            flag = 3
        elif (x_tip is not None) and (y_tip is not None) and (1310 <= x_tip <= 1910) and (750 <= y_tip <= 900):
            flag = 4
        elif (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (50 <= y_tip <= 200):
            flag = 5
    res_image = cv2.cvtColor(flippedRGB,
                             cv2.COLOR_RGB2BGR)  # вносим эти прямоугольники на изображение, котрое потом будем показывать
    cv2.putText(res_image, 'Immortal snake', (40, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                6)  # печатаем текст на кнопках
    cv2.putText(res_image, 'Snake Ninja', (85, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Fast snake', (1405, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Greedy snake', (1385, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Mortal snake', (735, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.imshow('Game', res_image)  # отображаем картинку
    if cv2.waitKey(1) == 65:
        return


def end(cap, handsDetector):  # функция завершения режимов с отображением результата и возможностью вернуться в меню
    global scr
    global mode
    global status
    global snake
    global apple_x
    global apple_y
    global center
    global flag
    global x_tip
    global y_tip
    global score
    ret, frame = cap.read()  # захватываем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    scr= 0  # обнуляем доп переменные и списки
    snake = []
    apple_x, apple_y, center = None, None, None
    cv2.rectangle(flippedRGB, (660, 850), (1260, 1000), (154, 214, 143),
                  -1)  # рисуем фон для текста и прямоугольник-кнопку для возвращения в меню
    cv2.rectangle(flippedRGB, (610, 50), (1310, 200), (255, 255, 255), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на картинке
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 15, (0, 0, 255), -1)  # рисуем кончик указательного пальца
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на изображении
    if status == 'normal':  # проверяем если мы не проиграли, то печатаем радостный текст
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255),
                    6)  # печатаем текст
    else:  # если мы проиграли
        cv2.putText(res_image, 'Game over((', (730, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255),
                    6)  # печатаем текст
    cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                6)  # печатаем текст для кнопки
    if (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (
            850 <= y_tip <= 1000):  # если нажали на возвращение в меню, то обнуляем остальные переменные
        flag = 0
        score = 0
        mode = 0
        x_tip, y_tip = None, None
        status = 'normal'
    cv2.imshow('Game', res_image)  # показываем изображение


def immortal_snake(cap, handsDetector, crit_dist, max_score):  # режим бессмертной змейки
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global status
    global flag
    global x_tip
    global y_tip
    ret, frame = cap.read()  # захватываем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape  # вычисляем размеры экрана
    if score == max_score:  # если набрали необходмое количество очков, то запускаем окончание
        end(cap, handsDetector)
    else:
        if apple_x is None or apple_y is None:  # создаем новое рандомное яблочко
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)
        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)  # рисуем фон для текста
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)  # рисуем яблоко
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на картинке
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:  # добавляем новый шарик если это только начало
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):  # сдвигаемся на один
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):  # отрисовываем змейку
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # опять показываем изображение
            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:  # смотрим собрали ли мы яблоко
                score += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])  # добавляем кружок в змейку

        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                    6)  # печатаем текст

        cv2.imshow('Game', res_image)  # показываем все

    if cv2.waitKey(1) == 65:
        return


def snake_ninja(cap, handsDetector, crit_dist, max_score, crit_time):  # режим змейки ниндзя
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global flag
    global x_tip
    global status
    global y_tip
    global start_time
    ret, frame = cap.read()  # захватываем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape  # находим размеры изображения
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if mode == 0:
        choose_mode(cap, handsDetector)  # выбираем сложность
    else:
        if score == max_score:  # если набрали нужное количество очков завершаем режим
            end(cap, handsDetector)
        else:
            x = time.time()  # сколько сейчас время
            if x - start_time >= crit_time:  # смотрим сколько времени прошло с момента когда мы последний раз создавали яблоко
                score = 0  # если много, то обнуляем счет и другие переменные
                scr, list_capacity = 0, 0
                snake = []
            if apple_x is None or apple_y is None or (
                    x - start_time >= crit_time):  # создаем яблоко с произвольными координаты
                apple_x = np.random.randint(30, width - 30)
                apple_y = np.random.randint(30, height - 30)
                start_time = time.time()  # записываем время когда это произошло
            apple = (apple_x, apple_y)
            cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)  # рисуем фон для текста
            cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)  # рисуем яблоко
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на картинке
            if results.multi_hand_landmarks is not None:
                # нас интересует только подушечка указательного пальца (индекс 8)
                # нужно умножить координаты а размеры картинки
                x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                            flippedRGB.shape[1])
                y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                            flippedRGB.shape[0])
                if len(snake) == 0:  # добавояем в змейку круг если она пуста
                    snake.append([x_tip, y_tip])
                snake[0][0] = x_tip
                snake[0][1] = y_tip
                if scr == 1:
                    for i in range(len(snake) - 1, 0, -1):  # сдвигаем все на один
                        snake[i][0] = snake[i - 1][0]
                        snake[i][1] = snake[i - 1][1]
                    scr = 0
                scr += 1
                for i in range(1, len(snake)):  # рисуем змейку
                    cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем на картинке
                center = (x_tip, y_tip)
                if dist(apple, center) < crit_dist:  # проверяем схватили ли мы яблоко
                    score += 1
                    apple_x = None
                    apple_y = None
                    snake.append([x_tip, y_tip])  # добавляем новый круг

            cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                        6)  # печатаем текст
            cv2.imshow('Game', res_image)  # отображаем все

    if cv2.waitKey(1) == 65:
        return


def snake_speedrunner(cap, handsDetector, crit_dist, max_score, crit_time):  # режим быстрой змейки
    global apple_x
    global apple_y
    global status
    global center
    global snake
    global score
    global scr
    global flag
    global x_tip
    global y_tip
    global start_time
    global mode
    ret, frame = cap.read()  # закват кадра
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape  # размеры окна
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if mode == 0:
        choose_mode(cap, handsDetector)  # выбираем сложность
    else:
        if score == max_score:  # если набрали нужное количество очков, то завершаем режим
            end(cap, handsDetector)
        else:
            x = time.time()  # смотрим какое сейчас время
            if apple_x is None or apple_y is None or (
                    x - start_time >= crit_time):  # если больше чем можно, то делаем новое яблоко
                apple_x = np.random.randint(30, width - 30)
                apple_y = np.random.randint(30, height - 30)
                start_time = time.time()  # смотрим когда создали это яблоко
            apple = (apple_x, apple_y)
            cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)  # рисуем фон для текста
            cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)  # рисуем яблоко
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на картинке
            if results.multi_hand_landmarks is not None:
                # нас интересует только подушечка указательного пальца (индекс 8)
                # нужно умножить координаты а размеры картинки
                x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                            flippedRGB.shape[1])
                y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                            flippedRGB.shape[0])
                if len(snake) == 0:  # создаем круг в змейке если она пустая
                    snake.append([x_tip, y_tip])
                snake[0][0] = x_tip
                snake[0][1] = y_tip
                if scr == 1:
                    for i in range(len(snake) - 1, 0, -1):  # сдвигаем все
                        snake[i][0] = snake[i - 1][0]
                        snake[i][1] = snake[i - 1][1]
                    scr = 0
                scr += 1
                for i in range(1, len(snake)):  # рисуем змейку
                    cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем все на картинке

                center = (x_tip, y_tip)
                if dist(apple, center) < crit_dist:  # смотрим съели ли мы яблоко
                    score += 1
                    apple_x = None
                    apple_y = None
                    snake.append([x_tip, y_tip])  # добавляем круг в змейку

            cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                        6)  # печатаем текст
            cv2.imshow('Game', res_image)  # отображаем изображение со всем этим

    if cv2.waitKey(1) == 65:
        return


def greedy_snake(cap, handsDetector, crit_dist, max_time):  # режим жадной змейки
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global flag
    global x_tip
    global y_tip
    global start_time
    global maxscore
    global status
    ret, frame = cap.read()  # захват кадра
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape  # размеры окна
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    x = time.time()  # сколько сейчас времени
    if (x - start_time) >= max_time:  # если больше чем нужно заканчиваем режим
        scr = 0  # обнуляем часть переменных
        snake = []
        apple_x, apple_y, center = None, None, None
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.rectangle(flippedRGB, (660, 850), (1260, 1000), (154, 214, 143),
                      -1)  # рисуем фон для текста и прямоугольник-кноку
        cv2.rectangle(flippedRGB, (610, 50), (1310, 200), (255, 255, 255), -1)
        cv2.rectangle(flippedRGB, (50, 250), (1870, 400), (255, 255, 255), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это на изображении
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            cv2.circle(flippedRGB, (x_tip, y_tip), 15, (0, 0, 255), -1)  # рисуем кончик указательного пальца
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это
        maxscore = max(maxscore, score)  # смотрим максимальное количество полученных очков
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255),
                    6)  # печатаем текст
        cv2.putText(res_image, 'Your score is ' + str(score) + ". The maximum score is " + str(maxscore) + ".",
                    (200, 345), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        if (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (
                850 <= y_tip <= 1000):  # проверяем нажали ли мы на кнопку возвращения в меню
            flag = 0
            score = 0
            x_tip, y_tip = None, None
            start_time = time.time()
        cv2.imshow('Game', res_image)  # показываем это все
    else:
        if apple_x is None or apple_y is None:  # рисуем яблоко
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)
            if score == 0:  # засекаем момент когда отрисовали первое яблоко
                start_time = time.time()
        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)  # рисуем фон для текста
        cv2.rectangle(flippedRGB, (1370, 50), (1800, 150), (255, 255, 255), -1)
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)  # печатаем текст
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это все
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:  # добавляем кружок в пустую змейку
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):  # сдвигаем на один
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):  # отрисовываем змейку
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это все
            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:  # проверяем взяли ли мы яблоко
                score += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])  # добавляем новое яблоко
        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                    6)  # печатаем текст
        cv2.putText(res_image, 'Time: ' + str(round(x - start_time, 2)), (1380, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                    (0, 0, 0), 6)
        cv2.imshow('Game', res_image)  # показываем это все

    if cv2.waitKey(1) == 65:
        return


def mortal_snake(cap, handsDetector, crit_dist, max_score):  # режим смертной змейки
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global flag
    global x_tip
    global y_tip
    global status
    ret, frame = cap.read()  # захватываем кадр
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape  # берем размер окна
    if score == max_score or status == 'game over':  # смотрим набрали ли мы нужное количество очков или умерли
        end(cap, handsDetector)
    else:
        if apple_x is None or apple_y is None:  # создаем новое яблоко с произвольными координатами
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)
        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)  # рисуем фон для текста
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)  # рисуем яблоко
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это все на картинке
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:  # добавляем новый круг в змейку
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):  # сдвигаем все на один
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):  # рисуем змейку
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)  # отображаем это все на картинке
            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:  # смотрим поймали ли мы яблоко
                score += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])  # добавляем новый круг в змейку

            if (10 >= x_tip) or (x_tip + 10 >= width) or (y_tip <= 10) or (
                    y_tip + 10 >= height):  # проверяем если коснулись границ поля, то умираем
                status = 'game over'
            snake2 = []
            for i in range(len(snake) - 1):  # делаем змейку набором отрезков (ломанной)
                v = [snake[i][0], snake[i][1], snake[i + 1][0], snake[i + 1][1]]
                snake2.append(v)
            for i in range(len(snake2)):
                for j in range(i + 1, len(snake2)):
                    p1 = [snake2[i][0], snake2[i][1]]
                    p2 = [snake2[i][2], snake2[i][3]]
                    p3 = [snake2[j][0], snake2[j][1]]
                    p4 = [snake2[j][2], snake2[j][3]]
                    if not (
                            p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4):  # если это два отрезка без общего начала, проверяем на пересечение
                        if snake_crossing(snake2[i], snake2[j]) == "YES":  # если пересеклась, змейка умирает
                            status = 'game over'
                            break
        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0),
                    6)  # печатаем текст
        cv2.imshow('Game', res_image)  # выводи все это

    if cv2.waitKey(1) == 65:
        return


cap = cv2.VideoCapture(0)

handsDetector = mp.solutions.hands.Hands()

while True:  # главный цикл игры
    if flag == 0:  # проверяем флаг, если равен нулю - идем в меню
        menu(cap, handsDetector)
    elif flag == 2:  # если равен 2 - идем в бессмертную змейку
        immortal_snake(cap, handsDetector, crit_dist, max_score)
    elif flag == 1:  # если равен 1 - идем в змейку-ниндзя
        snake_ninja(cap, handsDetector, crit_dist, max_score, crit_time)
    elif flag == 3:  # если равен 3 - идем в быструю змейку
        snake_speedrunner(cap, handsDetector, crit_dist, max_score, crit_time)
    elif flag == 4:  # если равен 4 - идем в жадную змейку
        greedy_snake(cap, handsDetector, crit_dist, max_time)
    elif flag == 5:  # если равен 5 - идем в смертную змейку
        mortal_snake(cap, handsDetector, crit_dist, max_score)

handsDetector.close()
cv2.destroyAllWindows()
cap.release()
cv2.waitKey(0)
