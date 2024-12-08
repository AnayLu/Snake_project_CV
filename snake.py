import cv2
import numpy as np
import mediapipe as mp
import time
import math

score = 0
max_score = 6
list_capacity = 0
max_lc = 20
crit_dist = 35
l = []
flag = 0
apple_radius = 10
apple_x, apple_y, center = None, None, None
snake = []
scr = 0
x_tip, y_tip = None, None
crit_time = 2
start_time = time.time()
max_time = 15
maxscore = 0
status = 'normal'
snake2 = []
mode = 0


class Point:
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


class Vector(Point):
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

def snake_crossing(v1, v2): #функция для определения пересечения отрезков, код взят у Дарьи Порай
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
        if (Vector(A, B) ^ Vector(A, Point(v2[2], v2[3]))) == 0 and (Vector(A, B) ^ Vector(A, Point(v2[0], v2[1]))) == 0:
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
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def choose_mode(cap, handsDetector):
    global mode
    global crit_time
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    cv2.rectangle(flippedRGB, (10, 750), (610, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (1310, 750), (1910, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (660, 750), (1260, 900), (154, 214, 143), -1)
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
        if (x_tip is not None) and (y_tip is not None) and (10<=x_tip<=610) and (750<=y_tip<=900):
            mode = 1
            crit_time = 6
        elif (x_tip is not None) and (y_tip is not None) and (660<=x_tip<=1260) and (750<=y_tip<=900):
            mode = 2
            crit_time = 4
        elif (x_tip is not None) and (y_tip is not None) and (1310<=x_tip<=1910) and (750<=y_tip<=900):
            mode = 3
            crit_time = 2
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.putText(res_image, 'Easy', (230, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Medium', (830, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Hard', (1530, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.imshow('Game', res_image)
    if cv2.waitKey(1) == 65:
        return
def menu(cap, handsDetector):
    global flag
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    cv2.rectangle(flippedRGB, (10, 750), (610, 900), (154, 214, 143), -1)
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
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)

        if (x_tip is not None) and (y_tip is not None) and(10<=x_tip<=610) and (750<=y_tip<=900):
            flag = 2
        elif (x_tip is not None) and (y_tip is not None) and(10<=x_tip<=610) and (350<=y_tip<=500):
            flag=1
        elif (x_tip is not None) and (y_tip is not None) and (1310 <= x_tip <= 1910) and (350 <= y_tip <= 500):
            flag = 3
        elif (x_tip is not None) and (y_tip is not None) and (1310 <= x_tip <= 1910) and (750<=y_tip<=900):
            flag = 4
        elif (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (50<=y_tip<=200):
            flag = 5
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.putText(res_image, 'Immortal snake', (40, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Snake Ninja', (85, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Fast snake', (1405, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Greedy snake', (1385, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Mortal snake', (735, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.imshow('Game', res_image)
    if cv2.waitKey(1) == 65:
        return

def end(cap, handsDetector):
    global scr
    global mode
    global status
    global list_capacity
    global snake
    global apple_x
    global apple_y
    global center
    global flag
    global x_tip
    global y_tip
    global score
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    scr, list_capacity = 0, 0
    snake = []
    apple_x, apple_y, center = None, None, None
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.rectangle(flippedRGB, (660, 850), (1260, 1000), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (610, 50), (1310, 200), (255, 255, 255), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 15, (0, 0, 255), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if status == 'normal':
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
    else:
        cv2.putText(res_image, 'Game over((', (730, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
    cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    if (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (850 <= y_tip <= 1000):
        flag = 0
        score = 0
        mode = 0
        x_tip, y_tip = None, None
        status = 'normal'
    cv2.imshow('Game', res_image)
def immortal_snake(cap, handsDetector, crit_dist, max_score):
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global status
    global list_capacity
    global flag
    global x_tip
    global y_tip
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    if score == max_score:
        end(cap, handsDetector)
    else:
        if apple_x is None or apple_y is None:
            # assigning random coefficients for apple coordinates
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)

        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)



            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:
                score += 1
                list_capacity += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])


        for i in range(1, len(l)):
            if l[i - 1] is None or l[i] is None:
                continue
            r, g, b = np.random.randint(0, 255, 3)

            cv2.line(res_image, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)
        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)

        cv2.imshow('Game', res_image)


    if cv2.waitKey(1) == 65:
        return

def snake_ninja(cap, handsDetector, crit_dist, max_score, crit_time):
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global list_capacity
    global flag
    global x_tip
    global status
    global y_tip
    global start_time
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if mode == 0:
        choose_mode(cap, handsDetector)
    else:
        if score == max_score:
            end(cap, handsDetector)
        else:
            x = time.time()
            if x- start_time>=crit_time:
                score = 0
                scr, list_capacity = 0, 0
                snake = []
            if apple_x is None or apple_y is None or (x-start_time >= crit_time):
                # assigning random coefficients for apple coordinates
                apple_x = np.random.randint(30, width - 30)
                apple_y = np.random.randint(30, height - 30)
                start_time = time.time()

            apple = (apple_x, apple_y)
            cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)
            cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks is not None:
                # нас интересует только подушечка указательного пальца (индекс 8)
                # нужно умножить координаты а размеры картинки
                x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                            flippedRGB.shape[1])
                y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                            flippedRGB.shape[0])
                if len(snake) == 0:
                    snake.append([x_tip, y_tip])
                snake[0][0] = x_tip
                snake[0][1] = y_tip
                if scr == 1:
                    for i in range(len(snake) - 1, 0, -1):
                        snake[i][0] = snake[i - 1][0]
                        snake[i][1] = snake[i - 1][1]
                    scr = 0
                scr += 1
                for i in range(1, len(snake)):
                    cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)



                center = (x_tip, y_tip)
                if dist(apple, center) < crit_dist:
                    score += 1
                    list_capacity += 1
                    apple_x = None
                    apple_y = None
                    snake.append([x_tip, y_tip])

            for i in range(1, len(l)):
                if l[i - 1] is None or l[i] is None:
                    continue
                r, g, b = np.random.randint(0, 255, 3)

                cv2.line(res_image, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)
            cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)

            cv2.imshow('Game', res_image)

    if cv2.waitKey(1) == 65:
        return

def snake_speedrunner(cap, handsDetector, crit_dist, max_score, crit_time):
    global apple_x
    global apple_y
    global status
    global center
    global snake
    global score
    global scr
    global list_capacity
    global flag
    global x_tip
    global y_tip
    global start_time
    global mode
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if mode == 0:
        choose_mode(cap, handsDetector)
    else:
        if score == max_score:
            end(cap, handsDetector)
        else:
            x = time.time()
            if apple_x is None or apple_y is None or (x - start_time >= crit_time):
                # assigning random coefficients for apple coordinates
                apple_x = np.random.randint(30, width - 30)
                apple_y = np.random.randint(30, height - 30)
                start_time = time.time()

            apple = (apple_x, apple_y)
            cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)
            cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks is not None:
                # нас интересует только подушечка указательного пальца (индекс 8)
                # нужно умножить координаты а размеры картинки
                x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                            flippedRGB.shape[1])
                y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                            flippedRGB.shape[0])
                if len(snake) == 0:
                    snake.append([x_tip, y_tip])
                snake[0][0] = x_tip
                snake[0][1] = y_tip
                if scr == 1:
                    for i in range(len(snake) - 1, 0, -1):
                        snake[i][0] = snake[i - 1][0]
                        snake[i][1] = snake[i - 1][1]
                    scr = 0
                scr += 1
                for i in range(1, len(snake)):
                    cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
                cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

                center = (x_tip, y_tip)
                if dist(apple, center) < crit_dist:
                    score += 1
                    list_capacity += 1
                    apple_x = None
                    apple_y = None
                    snake.append([x_tip, y_tip])

            for i in range(1, len(l)):
                if l[i - 1] is None or l[i] is None:
                    continue
                r, g, b = np.random.randint(0, 255, 3)

                cv2.line(res_image, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)
            cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)

            cv2.imshow('Game', res_image)

    if cv2.waitKey(1) == 65:
        return

def greedy_snake(cap, handsDetector, crit_dist, max_time):
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global list_capacity
    global flag
    global x_tip
    global y_tip
    global start_time
    global maxscore
    global status
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    x = time.time()
    if (x-start_time) >= max_time:
        scr, list_capacity = 0, 0
        snake = []
        apple_x, apple_y, center = None, None, None
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        cv2.rectangle(flippedRGB, (660, 850), (1260, 1000), (154, 214, 143), -1)
        cv2.rectangle(flippedRGB, (610, 50), (1310, 200), (255, 255, 255), -1)
        cv2.rectangle(flippedRGB, (50, 250), (1870, 400), (255, 255, 255), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            cv2.circle(flippedRGB, (x_tip, y_tip), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        maxscore = max(maxscore, score)
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Your score is ' + str(score) + ". The maximum score is " + str(maxscore) + ".", (200, 345), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        if (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (850 <= y_tip <= 1000):
            flag = 0
            score = 0
            x_tip, y_tip = None, None
            start_time = time.time()
        cv2.imshow('Game', res_image)
    else:
        if apple_x is None or apple_y is None:
            # assigning random coefficients for apple coordinates
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)
            if score == 0:
                start_time = time.time()

        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)
        cv2.rectangle(flippedRGB, (1370, 50), (1800, 150), (255, 255, 255), -1)
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:
                score += 1
                list_capacity += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])

        for i in range(1, len(l)):
            if l[i - 1] is None or l[i] is None:
                continue
            r, g, b = np.random.randint(0, 255, 3)

            cv2.line(res_image, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)
        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        cv2.putText(res_image, 'Time: ' + str(round(x-start_time, 2)), (1380, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)

        cv2.imshow('Game', res_image)

    if cv2.waitKey(1) == 65:
        return

def mortal_snake(cap, handsDetector, crit_dist, max_score):
    global apple_x
    global apple_y
    global center
    global snake
    global score
    global scr
    global list_capacity
    global flag
    global x_tip
    global y_tip
    global status
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    if score == max_score or status == 'game over':
        end(cap, handsDetector)
    else:
        if apple_x is None or apple_y is None:
            # assigning random coefficients for apple coordinates
            apple_x = np.random.randint(30, width - 30)
            apple_y = np.random.randint(30, height - 30)

        apple = (apple_x, apple_y)
        cv2.rectangle(flippedRGB, (50, 50), (400, 150), (255, 255, 255), -1)
        cv2.circle(flippedRGB, apple, apple_radius, (0, 255, 0), -1)
        res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:
            # нас интересует только подушечка указательного пальца (индекс 8)
            # нужно умножить координаты а размеры картинки
            x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                        flippedRGB.shape[1])
            y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                        flippedRGB.shape[0])
            if len(snake) == 0:
                snake.append([x_tip, y_tip])
            snake[0][0] = x_tip
            snake[0][1] = y_tip
            if scr == 1:
                for i in range(len(snake) - 1, 0, -1):
                    snake[i][0] = snake[i - 1][0]
                    snake[i][1] = snake[i - 1][1]
                scr = 0
            scr += 1
            for i in range(1, len(snake)):
                cv2.circle(flippedRGB, (snake[i][0], snake[i][1]), 15, (0, 0, 255), -1)
                res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
            cv2.circle(flippedRGB, (snake[0][0], snake[0][1]), 15, (0, 0, 255), -1)
            res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

            center = (x_tip, y_tip)
            if dist(apple, center) < crit_dist:
                score += 1
                list_capacity += 1
                apple_x = None
                apple_y = None
                snake.append([x_tip, y_tip])

            if (10>=x_tip) or (x_tip+10>=width) or (y_tip<=10) or (y_tip+10>=height):
                status = 'game over'
            snake2 = []
            for i in range(len(snake)):
                for j in range(i+1, len(snake)):
                    v = [snake[i][0], snake[i][1], snake[j][0], snake[j][1]]
                    snake2.append(v)
            for i in range(len(snake2)):
                for j in range(i+1, len(snake2)):
                    p1 = [snake2[i][0], snake2[i][1]]
                    p2 = [snake2[i][2], snake2[i][3]]
                    p3 = [snake2[j][0], snake2[j][1]]
                    p4 = [snake2[j][2], snake2[j][3]]
                    if not (abs(p1[0]-p3[0])<=20 or abs(p1[1]-p3[1])<=20 or abs(p1[0]-p4[0])<=20 or abs(p1[1]-p4[1])<=20 or abs(p2[0]-p3[0])<=20 or abs(p2[1]-p3[1])<=20 or abs(p2[0]-p4[0])<=20 or abs(p2[1]-p4[1])<=20):
                        if snake_crossing(snake2[i], snake2[j]) == "YES":
                            status = 'game over'
                            print(snake2[i], snake2[j])
                            break

        for i in range(1, len(l)):
            if l[i - 1] is None or l[i] is None:
                continue
            r, g, b = np.random.randint(0, 255, 3)

            cv2.line(res_image, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)
        cv2.putText(res_image, 'Score: ' + str(score), (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)

        cv2.imshow('Game', res_image)

    if cv2.waitKey(1) == 65:
        return

cap = cv2.VideoCapture(0)

handsDetector = mp.solutions.hands.Hands()

while True:
    if flag == 0:
        menu(cap, handsDetector)
    elif flag == 2:
        immortal_snake(cap, handsDetector, crit_dist, max_score)
    elif flag == 1:
        snake_ninja(cap, handsDetector, crit_dist, max_score, crit_time)
    elif flag == 3:
        snake_speedrunner(cap, handsDetector, crit_dist, max_score, crit_time)
    elif flag == 4:
        greedy_snake(cap, handsDetector, crit_dist, max_time)
    elif flag == 5:
        mortal_snake(cap, handsDetector, crit_dist, max_score)

handsDetector.close()
cv2.destroyAllWindows()
cap.release()