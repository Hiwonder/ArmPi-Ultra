import math
import numpy as np

def calculate_e_distance(point1, point2):
    e_distance = int(round(math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))))
    return e_distance

def position_change_or_not(last_point, current_points, distance):
    for p in current_points:
        if last_point[0][:-1] == p[0][:-1]:
            dis = calculate_e_distance(last_point[1], p[1])
            if dis < distance:
                current_points.remove(p)
                p[0] = last_point[0]
                return False, p, current_points
    
    return True, None, current_points
    
def position_reorder(current_points, last_points, distance=10):
    new_points = []
    haved_change_points = []
    for p in last_points:
        res, not_change_point, haved_change_points = position_change_or_not(p, current_points, distance)
        if not res:
            new_points.extend([not_change_point])
    if haved_change_points != [] and new_points != []:
        names = np.array(new_points, dtype=object)[:, 0].tolist()
        for p in haved_change_points:
            index = 0
            while True:
                new_name = p[0][:-1]
                index += 1
                new_name += str(index)
                if new_name not in names:
                    p[0] = new_name
                    new_points.extend([p])
                    names.append(new_name)
                    break

    return new_points
