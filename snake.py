import cv2
import numpy as np
import mediapipe as mp
import time

score = 0
max_score = 11
list_capacity = 0
max_lc = 20
crit_dist = 25
l = []
flag = 0
apple_radius = 10
apple_x, apple_y, center = None, None, None
snake = []
scr = 0
x_tip, y_tip = None, None
crit_time = 2
start_time = time.time()

# distance function
def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def menu(cap, handsDetector):
    global flag
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    cv2.rectangle(flippedRGB, (10, 750), (610, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (660, 750), (1260, 900), (154, 214, 143), -1)
    cv2.rectangle(flippedRGB, (1310, 750), (1910, 900), (154, 214, 143), -1)
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
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.putText(res_image, 'Immortal snake', (40, 845), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Snake Ninja', (85, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.putText(res_image, 'Fast snake', (1405, 445), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
    cv2.imshow('live feed', res_image)
    if cv2.waitKey(1) == 65:
        return

def immortal_snake(cap, handsDetector, crit_dist, max_score):
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
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    if score == max_score:
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
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        if (x_tip is not None) and (y_tip is not None) and (660<=x_tip<=1260) and (850<=y_tip<=1000):
            flag = 0
            score = 0
            x_tip, y_tip = None, None
        cv2.imshow('live feed', res_image)
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

        cv2.imshow('live feed', res_image)


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
    global y_tip
    global start_time
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if score == max_score:
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
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        if (x_tip is not None) and (y_tip is not None) and (660<=x_tip<=1260) and (850<=y_tip<=1000):
            flag = 0
            score = 0
            x_tip, y_tip = None, None
        cv2.imshow('live feed', res_image)
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

        cv2.imshow('live feed', res_image)

    if cv2.waitKey(1) == 65:
        return

def snake_speedrunner(cap, handsDetector, crit_dist, max_score, crit_time):
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
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    height, width, _ = frame.shape
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    if score == max_score:
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
        cv2.putText(res_image, 'Congratulations!!', (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(res_image, 'Back to menu', (710, 945), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6)
        if (x_tip is not None) and (y_tip is not None) and (660 <= x_tip <= 1260) and (850 <= y_tip <= 1000):
            flag = 0
            score = 0
            x_tip, y_tip = None, None
        cv2.imshow('live feed', res_image)
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

        cv2.imshow('live feed', res_image)

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

handsDetector.close()
cv2.destroyAllWindows()
cap.release()