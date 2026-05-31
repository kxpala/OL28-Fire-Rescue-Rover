"""
Fire-Rescue-Rover Controller
Platform: Pioneer 3-AT in Webots

Student: Krisha Pala (s5433753)
"""

from controller import Robot
import math
import heapq


# WEBOTS SETUP

robot = Robot()
timestep = int(robot.getBasicTimeStep())


# DEVICE NAMES

LEFT_MOTOR_NAMES = ["front left wheel", "back left wheel"]
RIGHT_MOTOR_NAMES = ["front right wheel", "back right wheel"]

CAMERA_NAMES = ["camera0"]
LED_NAMES = ["led0"]
GPS_NAMES = ["gps0"]
COMPASS_NAMES = ["compass0"]
LIDAR_NAMES = ["lidar0"]
SONAR_NAMES = ["so0", "so1", "so2", "so3", "so4", "so5", "so6", "so7"]


# CONSTANTS

MAX_SPEED = 6.0

# Motor speeds
FORWARD_SPEED = 0.20 * MAX_SPEED
TARGET_SPEED = 0.16 * MAX_SPEED
TURN_SPEED = 0.18 * MAX_SPEED
SCAN_SPEED = 0.12 * MAX_SPEED
REVERSE_SPEED = 0.22 * MAX_SPEED

# Red target detection thresholds.
RED_LOCK_THRESHOLD = 0.0015
RED_CLOSE_THRESHOLD = 0.45
RED_CONFIRM_FRAMES = 3
RED_DEADZONE = 0.16

# Green exit/home beacon thresholds.
GREEN_DETECT_THRESHOLD = 0.0005
GREEN_COMPLETE_THRESHOLD = 0.08

# Target approach and recovery timing.
TARGET_MEMORY_TIME = 8.0
REACQUIRE_TIME = 4.0
TARGET_CHASE_FRONT_CLEARANCE = 0.50
TARGET_RETRIEVAL_DISTANCE = 0.12
TARGET_APPROACH_STOP_DISTANCE = 0.50

# Obstacle avoidance.
OBSTACLE_FRONT_DISTANCE = 0.45
OBSTACLE_SIDE_DISTANCE = 0.30
RETURN_FRONT_AVOID_DISTANCE = 0.55


# LIDAR occupancy grid settings.
GRID_SIZE = 120
CELL_SIZE = 0.10
MAP_RANGE_LIMIT = 4.0
OCCUPIED = 1
FREE = 0

# Home/exit navigation.
HOME_DISTANCE = 0.08
WAYPOINT_DISTANCE = 0.22
PLAN_EVERY = 1.0
MIN_EXIT_TIME = 2.0
EXIT_BACKUP_TIME = 1.2
EXIT_TURN_TIME = 2.0
EXIT_CLEAR_TIME = 1.5

# Stuck/failsafe behaviour.
STUCK_TIME = 3.0
MIN_MOVE_DISTANCE = 0.025
FAILSAFE_REVERSE_TIME = 1.8
FAILSAFE_TURN_TIME = 2.4
FAILSAFE_IGNORE_TARGET_TIME = 5.0

# Timed local obstacle avoidance.
AVOID_REVERSE_TIME = 1.0
AVOID_TOTAL_TIME = 2.6


# FSM STATES

EXPLORE = "EXPLORE"
TARGET_DETECTION = "TARGET_DETECTION"
NAVIGATE_TO_TARGET = "NAVIGATE_TO_TARGET"
RETRIEVAL_CONFIRMATION = "RETRIEVAL_CONFIRMATION"
EXIT_NAVIGATION = "EXIT_NAVIGATION"
FAILSAFE = "FAILSAFE"
MISSION_COMPLETE = "MISSION_COMPLETE"

state = EXPLORE



# DEVICE HELPERS

def get_device(name):
    """Safely fetch a Webots device. Returns None instead of crashing if not found."""
    try:
        return robot.getDevice(name)
    except Exception:
        return None


def get_first_device(names):
    """Try a list of possible device names and return the first one that exists."""
    for name in names:
        device = get_device(name)
        if device is not None:
            print("Using device:", name)
            return device
    return None


# Motors are split into left and right groups for 4-wheel differential drive.
left_motors = []
right_motors = []


for name in LEFT_MOTOR_NAMES:
    motor = get_device(name)
    if motor is not None:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        left_motors.append(motor)


for name in RIGHT_MOTOR_NAMES:
    motor = get_device(name)
    if motor is not None:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        right_motors.append(motor)

print("Left motors:", len(left_motors))
print("Right motors:", len(right_motors))


camera = get_first_device(CAMERA_NAMES)
if camera is not None:
    camera.enable(timestep)
else:
    print("WARNING: Camera not found.")


led = get_first_device(LED_NAMES)
if led is not None:
    led.set(0)
else:
    print("WARNING: LED not found. Console LED messages will be used instead.")


gps = get_first_device(GPS_NAMES)
if gps is not None:
    gps.enable(timestep)
else:
    print("WARNING: GPS not found.")


compass = get_first_device(COMPASS_NAMES)
if compass is not None:
    compass.enable(timestep)
else:
    print("WARNING: Compass not found.")


lidar = get_first_device(LIDAR_NAMES)
if lidar is not None:
    lidar.enable(timestep)
    try:
        lidar.enablePointCloud()
    except Exception:
        pass
else:
    print("WARNING: LIDAR not found.")


sonars = []
for name in SONAR_NAMES:
    sensor = get_device(name)
    if sensor is not None:
        sensor.enable(timestep)
        sonars.append(sensor)

print("Sonars:", len(sonars))



# GLOBAL MEMORY

home_position = None

# Target memory is used so a brief camera loss does not instantly fail the mission.
target_last_seen_time = 0.0
target_last_bearing = 0.0
red_seen_frames = 0
ignore_target_until = 0.0
reacquire_until = 0.0

# Occupancy grid and A* path memory.
grid = [[FREE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
current_path = []
last_plan_time = 0.0

# Stuck detection.
last_position = None
last_position_time = 0.0

# Failsafe and mission flags.
failsafe_start = 0.0
retrieval_done = False
exit_navigation_start = 0.0
exit_escape_start = 0.0 

# Timed obstacle avoidance memory.
avoid_turn_until = 0.0
avoid_active_until = 0.0
avoid_direction = "left"

# Console LED/indicator memory, used to avoid printing every timestep.
last_indicator_message = ""
last_indicator_time = 0.0


# ACTUATION / MOVEMENT

def clamp(value, low, high):
    """Limit a value between a minimum and maximum."""
    
    return max(low, min(high, value))


def set_motors(left_speed, right_speed):
    """Set all left wheels and all right wheels."""
    
    left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)

    for motor in left_motors:
        motor.setVelocity(left_speed)

    for motor in right_motors:
        motor.setVelocity(right_speed)


def stop_robot():
    """Stop all wheels."""
    set_motors(0.0, 0.0)


def reverse(speed=REVERSE_SPEED):
    """Reverse straight backwards."""
    set_motors(-speed, -speed)


def turn_left(speed=TURN_SPEED):
    """Rotate left in place."""
    set_motors(-speed, speed)


def turn_right(speed=TURN_SPEED):
    """Rotate right in place."""
    set_motors(speed, -speed)


def set_led(value):
    """Set the Webots LED if available."""
    
    if led is not None:
        led.set(value)


def mission_indicator(message, force=False):
    """Print a visible mission status message."""
    
    global last_indicator_message, last_indicator_time

    current_time = robot.getTime()
    if force or message != last_indicator_message or current_time - last_indicator_time > 2.0:
        print("[MISSION INDICATOR]", message)
        last_indicator_message = message
        last_indicator_time = current_time


# SENSING

def get_position():
    """Read GPS position. """
    
    if gps is None:
        return None

    values = gps.getValues()
    return values[0], values[1]


def get_heading():
    """Read compass heading in the X/Y floor plane for A* waypoint/path support."""
    
    if compass is None:
        return None

    north = compass.getValues()
    return math.atan2(north[0], north[1])


def normalise_angle(angle):
    """Convert an angle to the range -pi to +pi."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def read_sonar():
    """Return all sonar values as a list."""
    return [sensor.getValue() for sensor in sonars]


def sonar_front_obstacle():
    """Sonar fallback for close obstacle checks."""
    
    values = read_sonar()
    if len(values) >= 8:
        front_strength = max(values[1], values[2], values[3], values[4])
        return front_strength > 900, front_strength
    return False, 0.0


def detect_colour(target):
    """
    Generic colour detector for camera-based perception.

    target = "red" detects the retrieval object.
    target = "green" detects the exit/home beacon.

    Returns:
    - ratio: percentage of sampled image pixels matching the colour
    - bearing: horizontal offset from centre, where 0 means centred
    """
    
    if camera is None:
        return 0.0, 0.0

    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    if image is None or width <= 0 or height <= 0:
        return 0.0, 0.0

    colour_count = 0
    total_count = 0
    x_sum = 0

    for y in range(0, height, 2):
        for x in range(0, width, 2):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)

            total_count += 1

            if target == "red":
                match = r > 80 and r > g * 1.4 and r > b * 1.4
            elif target == "green":
                match = g > 80 and g > r * 1.4 and g > b * 1.4
            else:
                match = False

            if match:
                colour_count += 1
                x_sum += x

    if colour_count == 0:
        return 0.0, 0.0

    ratio = colour_count / total_count
    centre_x = x_sum / colour_count
    bearing = (centre_x - width / 2.0) / (width / 2.0)
    return ratio, bearing


def detect_red():
    """Detect red retrieval target only before retrieval."""
    
    if retrieval_done:
        return 0.0, 0.0

    return detect_colour("red")


def detect_green():
    """Detect green exit/home beacon."""
    return detect_colour("green")


def lidar_sector_distances():
    """Split LIDAR readings into front/left/right sectors. This is used by obstacle avoidance and return navigation."""
    
    if lidar is None:
        return 999.0, 999.0, 999.0

    try:
        ranges = lidar.getRangeImage()
        resolution = lidar.getHorizontalResolution()
        fov = lidar.getFov()
        
    except Exception:
        return 999.0, 999.0, 999.0

    if not ranges or resolution <= 0:
        return 999.0, 999.0, 999.0

    front = 999.0
    left = 999.0
    right = 999.0

    for i, distance in enumerate(ranges):
        if math.isinf(distance) or math.isnan(distance):
            continue

        angle = -fov / 2.0 + (i / max(1, resolution - 1)) * fov

        if abs(angle) < math.radians(22):
            front = min(front, distance)
        elif math.radians(22) <= angle <= math.radians(85):
            left = min(left, distance)
        elif math.radians(-85) <= angle <= math.radians(-22):
            right = min(right, distance)

    return front, left, right


def front_is_clear_for_target():
    """Check if the rover can safely approach the red target."""
    
    front, _, _ = lidar_sector_distances()
    return front > TARGET_CHASE_FRONT_CLEARANCE


def target_is_close_enough():
    """Confirm physical closeness to the red target using LIDAR only."""
    
    front, _, _ = lidar_sector_distances()
    return front < TARGET_RETRIEVAL_DISTANCE


def target_approach_obstacle_blocked():
    """During target approach, allow the robot to get close to the red object."""
    
    front, _, _ = lidar_sector_distances()

    if front < TARGET_APPROACH_STOP_DISTANCE:
        return True

    values = read_sonar()
    if len(values) >= 8:
        front_strength = max(values[1], values[2], values[3], values[4])
        if front_strength > 980:
            return True

    return False


# LIDAR OCCUPANCY GRID + A* PATH PLANNING

def world_to_grid(x, y):
    """Convert Webots world coordinates to occupancy-grid coordinates."""
    gx = int(round(x / CELL_SIZE + GRID_SIZE / 2))
    gy = int(round(y / CELL_SIZE + GRID_SIZE / 2))
    return gx, gy


def grid_to_world(gx, gy):
    """Convert occupancy-grid coordinates back to Webots world coordinates."""
    x = (gx - GRID_SIZE / 2) * CELL_SIZE
    y = (gy - GRID_SIZE / 2) * CELL_SIZE
    return x, y


def in_grid(gx, gy):
    """Check if a grid cell is inside the occupancy grid."""
    return 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE


def mark_obstacle_with_padding(gx, gy, padding=1):
    """Mark an obstacle plus a safety margin."""
    
    for dy in range(-padding, padding + 1):
        for dx in range(-padding, padding + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny):
                grid[ny][nx] = OCCUPIED


def update_lidar_map():
    """
    Update the occupancy grid using LIDAR.

    This is the simplified SLAM-style mapping component:
    - GPS/compass provide localisation
    - LIDAR marks nearby obstacles in the map
    - A* can plan through this grid
    """
    
    if lidar is None:
        return

    position = get_position()
    heading = get_heading()

    if position is None or heading is None:
        return

    robot_x, robot_y = position

    try:
        ranges = lidar.getRangeImage()
        resolution = lidar.getHorizontalResolution()
        fov = lidar.getFov()
    except Exception:
        return

    if resolution <= 0 or not ranges:
        return

    rgx, rgy = world_to_grid(robot_x, robot_y)
    if in_grid(rgx, rgy):
        grid[rgy][rgx] = FREE

    for i, distance in enumerate(ranges):
        if distance is None or math.isinf(distance) or math.isnan(distance):
            continue

        if distance <= 0.05 or distance > MAP_RANGE_LIMIT:
            continue

        local_angle = -fov / 2.0 + (i / max(1, resolution - 1)) * fov
        world_angle = heading + local_angle

        obs_x = robot_x + distance * math.sin(world_angle)
        obs_y = robot_y + distance * math.cos(world_angle)

        gx, gy = world_to_grid(obs_x, obs_y)
        if in_grid(gx, gy):
            mark_obstacle_with_padding(gx, gy, padding=1)


def heuristic(a, b):
    """Distance heuristic used by A*."""
    
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cell_is_free(cell):
    """Check whether a grid cell is available for path planning."""
    
    gx, gy = cell
    return in_grid(gx, gy) and grid[gy][gx] != OCCUPIED


def neighbours(cell):
    """Return all valid neighbouring cells for A* expansion."""
    
    gx, gy = cell
    options = [
        (gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1),
        (gx + 1, gy + 1), (gx + 1, gy - 1), (gx - 1, gy + 1), (gx - 1, gy - 1)
    ]
    return [n for n in options if cell_is_free(n)]


def astar(start, goal):
    """
    A* path planning on the LIDAR occupancy grid.

    heapq is used as a priority queue so the lowest-cost candidate cell is
    expanded first.
    """
    if not cell_is_free(start):
        return []

    if in_grid(goal[0], goal[1]):
        grid[goal[1]][goal[0]] = FREE

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for n in neighbours(current):
            step_cost = 1.4 if n[0] != current[0] and n[1] != current[1] else 1.0
            tentative_g = g_score[current] + step_cost

            if n not in g_score or tentative_g < g_score[n]:
                came_from[n] = current
                g_score[n] = tentative_g
                f_score = tentative_g + heuristic(n, goal)
                heapq.heappush(open_set, (f_score, n))

    return []


def clear_area_around_cell(cell, radius=3):
    """Clear a small area around the robot/home to prevent self-blocking in the grid."""
    
    gx, gy = cell
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny):
                grid[ny][nx] = FREE


def reset_occupancy_grid():
    """Reset the map if it becomes too noisy or over-blocked."""
    
    global grid
    grid = [[FREE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def plan_path_to_home():
    """
    Plan an A* path to the saved starting waypoint.
    """
    
    global current_path, last_plan_time

    position = get_position()
    if position is None or home_position is None:
        current_path = []
        return

    start = world_to_grid(position[0], position[1])
    goal = world_to_grid(home_position[0], home_position[1])

    clear_area_around_cell(start, radius=5)
    clear_area_around_cell(goal, radius=5)

    current_path = astar(start, goal)

    if not current_path:
        print("A* first attempt failed. Resetting map and retrying.")
        reset_occupancy_grid()
        update_lidar_map()
        clear_area_around_cell(start, radius=5)
        clear_area_around_cell(goal, radius=5)
        current_path = astar(start, goal)

    last_plan_time = robot.getTime()
    print("A* path cells:", len(current_path), "distance home:", round(distance_to_home(), 2))


def distance_to_home():
    """Return GPS distance from current position to saved starting waypoint."""
    
    position = get_position()
    if position is None or home_position is None:
        return 999.0

    dx = home_position[0] - position[0]
    dy = home_position[1] - position[1]
    return math.sqrt(dx * dx + dy * dy)


# CONTROL HELPERS

def reset_stuck_timer():
    """Reset the timer used to detect if the rover is physically stuck."""
    
    global last_position, last_position_time
    last_position = get_position()
    last_position_time = robot.getTime()


def change_state(new_state):
    """
    Change FSM state and perform any entry actions.

    Entry actions include resetting stuck detection and planning a home path
    when entering EXIT_NAVIGATION.
    """
    
    global state, red_seen_frames, failsafe_start, current_path

    if state != new_state:
        print("STATE:", state, "->", new_state)
        state = new_state
        reset_stuck_timer()

        if new_state == EXPLORE:
            red_seen_frames = 0

        if new_state == FAILSAFE:
            failsafe_start = 0.0

        if new_state == EXIT_NAVIGATION:
            current_path = []
            plan_path_to_home()


def reactive_avoidance():
    """
    General obstacle avoidance used before target retrieval.

    Behaviour:
    1. If an obstacle is in front, reverse briefly.
    2. Then turn toward the clearer side.
    3. Return to the FSM state behaviour.
    """
    
    global avoid_turn_until, avoid_active_until, avoid_direction

    current_time = robot.getTime()
    front, left, right = lidar_sector_distances()

    if current_time < avoid_active_until:
        if current_time < avoid_turn_until:
            reverse(REVERSE_SPEED)
        else:
            if avoid_direction == "left":
                turn_left(TURN_SPEED)
            else:
                turn_right(TURN_SPEED)
        return True

    sonar_blocked, sonar_value = sonar_front_obstacle()

    if front < OBSTACLE_FRONT_DISTANCE or sonar_blocked:
        print("Avoiding obstacle. front:", round(front, 2), "sonar:", round(sonar_value, 1))

        if abs(left - right) < 0.05:
            avoid_direction = "left" if int(current_time) % 2 == 0 else "right"
        else:
            avoid_direction = "left" if left > right else "right"

        avoid_turn_until = current_time + AVOID_REVERSE_TIME
        avoid_active_until = current_time + AVOID_TOTAL_TIME
        reverse(REVERSE_SPEED)
        return True

    return False


def steer_to_red(bearing):
    """Steer toward the red target using camera bearing."""
    
    if abs(bearing) < RED_DEADZONE:
        set_motors(TARGET_SPEED, TARGET_SPEED)
        return

    turn = 1.8 * bearing
    left = TARGET_SPEED + turn
    right = TARGET_SPEED - turn
    set_motors(left, right)


def steer_to_green(bearing):
    """Steer toward the green exit beacon using camera bearing."""
    
    if abs(bearing) < RED_DEADZONE:
        set_motors(FORWARD_SPEED, FORWARD_SPEED)
        return

    turn = 1.8 * bearing
    left = FORWARD_SPEED + turn
    right = FORWARD_SPEED - turn
    set_motors(left, right)


def scan_to_reacquire_target():
    """Rotate in the last known target direction to reacquire the red object."""
    
    if target_last_bearing >= 0:
        turn_right(SCAN_SPEED)
    else:
        turn_left(SCAN_SPEED)


def check_stuck():
    """
    Detect if the robot is stuck. If GPS position barely changes over STUCK_TIME seconds, the rover enters FAILSAFE.
    """
    
    global last_position, last_position_time

    position = get_position()
    if position is None:
        return False

    current_time = robot.getTime()

    if last_position is None:
        last_position = position
        last_position_time = current_time
        return False

    if current_time - last_position_time < STUCK_TIME:
        return False

    moved = math.sqrt(
        (position[0] - last_position[0]) ** 2 +
        (position[1] - last_position[1]) ** 2
    )

    last_position = position
    last_position_time = current_time

    return moved < MIN_MOVE_DISTANCE


# FSM HANDLERS

def handle_explore():
    """
    EXPLORE:
    Default state. Rover drives forward, avoids obstacles, and searches for
    the red target with the camera.
    """
    global red_seen_frames, target_last_seen_time, target_last_bearing
    
    if retrieval_done:
        change_state(EXIT_NAVIGATION)
        return

    current_time = robot.getTime()

    if reactive_avoidance():
        if check_stuck():
            change_state(FAILSAFE)
        return

    ratio, bearing = detect_red()

    if current_time < ignore_target_until:
        set_motors(FORWARD_SPEED, FORWARD_SPEED)
        return

    # Multi-condition decision: red must be visible for several frames AND front must be clear.
    if ratio >= RED_LOCK_THRESHOLD and front_is_clear_for_target():
        red_seen_frames += 1
        target_last_seen_time = current_time
        target_last_bearing = bearing
    
    else:
        red_seen_frames = 0

    if red_seen_frames >= RED_CONFIRM_FRAMES:
        stop_robot()
        change_state(TARGET_DETECTION)
        return

    set_motors(FORWARD_SPEED, FORWARD_SPEED)

    if check_stuck():
        change_state(FAILSAFE)


def handle_target_detection():
    """
    TARGET_DETECTION:
    Transition state used to separate detection from approach.
    """
    if retrieval_done:
        change_state(EXIT_NAVIGATION)
        return
    
    stop_robot()
    change_state(NAVIGATE_TO_TARGET)


def handle_navigate_to_target():
    """
    NAVIGATE_TO_TARGET:
    Camera bearing steers the robot to the red object.
    - red must be clearly visible and large in the image
    - LIDAR must report that the rover is almost touching the object

    This prevents early retrieval confirmation from far away.
    """
    global target_last_seen_time, target_last_bearing, reacquire_until
    
    if retrieval_done:
        change_state(EXIT_NAVIGATION)
        return

    current_time = robot.getTime()
    ratio, bearing = detect_red()

    if ratio > 0.004:
        target_last_seen_time = current_time
        target_last_bearing = bearing

        print(
            "Target approach:",
            "red=", round(ratio, 3),
            "bearing=", round(bearing, 2),
            "front=", round(lidar_sector_distances()[0], 2)
        )

        # Multi-condition retrieval confirmation.
        if ratio >= RED_CLOSE_THRESHOLD and target_is_close_enough():
            stop_robot()
            change_state(RETRIEVAL_CONFIRMATION)
            return

        # If something is extremely close but red is not large enough yet, do not confirm retrieval. Stop briefly and let the FSM recover.
        if target_approach_obstacle_blocked():
            stop_robot()

            # Retrieval is confirmed when the red object is visually large and a close obstacle is detected.
            if ratio >= 0.40:
                print("Target touched. Confirming retrieval.")
                change_state(RETRIEVAL_CONFIRMATION)
                return

            print("Obstacle near target, but red is not clear enough.")
            change_state(FAILSAFE)
            return

        steer_to_red(bearing)
        return

    # If red is temporarily lost, search briefly before failing.
    if reactive_avoidance():
        if check_stuck():
            change_state(FAILSAFE)
        return

    if current_time - target_last_seen_time < TARGET_MEMORY_TIME:
        scan_to_reacquire_target()
        return

    if reacquire_until == 0.0:
        reacquire_until = current_time + REACQUIRE_TIME

    if current_time < reacquire_until:
        scan_to_reacquire_target()
        return

    reacquire_until = 0.0
    change_state(FAILSAFE)


def handle_retrieval_confirmation():
    """
    RETRIEVAL_CONFIRMATION:
    Simulates successful object retrieval. Since the physical LED was not
    visible in the robot model, the LED output is represented with a console
    indicator message.
    """
    global retrieval_done, exit_navigation_start, exit_escape_start

    stop_robot()
    set_led(1)

    retrieval_done = True
    exit_navigation_start = robot.getTime()
    exit_escape_start = robot.getTime() 

    mission_indicator("TARGET RETRIEVED - LED FLASHING NOW", force=True)
    print("Retrieval confirmed")

    change_state(EXIT_NAVIGATION)


def handle_exit_navigation():
    """
    Controls the robot after the red target has been retrieved.
    """
    current_time = robot.getTime()

    distance = distance_to_home()
    front, left, right = lidar_sector_distances()
    green_ratio, green_bearing = detect_green()

    if current_time - last_plan_time > PLAN_EVERY:
        plan_path_to_home()

    mission_indicator("LED FLASHING - RETURNING TO EXIT")

    print(
        "Return distance:", round(distance, 2),
        "front:", round(front, 2),
        "left:", round(left, 2),
        "right:", round(right, 2),
        "green:", round(green_ratio, 4),
        "A* cells:", len(current_path)
    )

    # Escape away from red first: reverse -> turn -> drive clear
    escape_elapsed = current_time - exit_escape_start

    if escape_elapsed < EXIT_BACKUP_TIME:
        mission_indicator("LED FLASHING - BACKING AWAY FROM TARGET")
        reverse(REVERSE_SPEED)
        return

    if escape_elapsed < EXIT_BACKUP_TIME + EXIT_TURN_TIME:
        mission_indicator("LED FLASHING - TURNING AWAY FROM TARGET")
        turn_left(TURN_SPEED)
        return

    if escape_elapsed < EXIT_BACKUP_TIME + EXIT_TURN_TIME + EXIT_CLEAR_TIME:
        mission_indicator("LED FLASHING - CLEARING TARGET AREA")
        if front > 0.45:
            set_motors(FORWARD_SPEED, FORWARD_SPEED)
        else:
            turn_left(TURN_SPEED)
        return

    # Mission complete when close to green/home
    if current_time - exit_navigation_start > MIN_EXIT_TIME:
        if distance < 0.05 or green_ratio > 0.18:
            stop_robot()
            set_led(1)

            print("MISSION COMPLETE")
            print("Target retrieved successfully")
            print("Returned safely to exit beacon")

            mission_indicator("MISSION COMPLETE - EXIT REACHED", force=True)
            change_state(MISSION_COMPLETE)
            return

    # If green is visible, chase it like red target. After retrieval, red is ignored and the green beacon becomes the active visual goal.
    if green_ratio > GREEN_DETECT_THRESHOLD:
        steer_to_green(green_bearing)
        return

    # Obstacle avoidance while searching for green
    if front <= RETURN_FRONT_AVOID_DISTANCE:
        if left > right:
            set_motors(-0.3 * FORWARD_SPEED, FORWARD_SPEED)
        else:
            set_motors(FORWARD_SPEED, -0.3 * FORWARD_SPEED)
        return

    if left < OBSTACLE_SIDE_DISTANCE:
        turn_right(TURN_SPEED)
        return

    if right < OBSTACLE_SIDE_DISTANCE:
        turn_left(TURN_SPEED)
        return

    # If green is not visible, scan for it
    turn_left(SCAN_SPEED)

    if check_stuck():
        change_state(FAILSAFE)


def handle_failsafe():
    """
    FAILSAFE:
    Safety recovery state.

    Used when the rover is stuck, target is lost for too long, and navigation cannot continue safely

    Behaviour: Reverse, Turn, Return to EXPLORE or EXIT_NAVIGATION depending on mission progress
    """
    global failsafe_start, ignore_target_until, reacquire_until

    current_time = robot.getTime()

    if failsafe_start == 0.0:
        failsafe_start = current_time
        print("Failsafe started.")

    elapsed = current_time - failsafe_start

    if elapsed < FAILSAFE_REVERSE_TIME:
        reverse(REVERSE_SPEED)
        return

    if elapsed < FAILSAFE_REVERSE_TIME + FAILSAFE_TURN_TIME:
        turn_left(TURN_SPEED)
        return

    ignore_target_until = current_time + FAILSAFE_IGNORE_TARGET_TIME
    reacquire_until = 0.0
    failsafe_start = 0.0

    if retrieval_done:
        change_state(EXIT_NAVIGATION)
    else:
        change_state(EXPLORE)


def handle_mission_complete():
    """MISSION_COMPLETE: Stop rover and keep final indicator active."""
    
    stop_robot()
    set_led(1)


# MAIN LOOP

print("Fire-Rescue-Rover controller started.")

while robot.step(timestep) != -1:
    if home_position is None:
        pos = get_position()
        if pos is not None:
            home_position = pos
            last_position = pos
            last_position_time = robot.getTime()
            if gps is not None:
                print("Raw GPS:", gps.getValues())
            print("Home/exit waypoint saved:", home_position)

    # Sense/update map every timestep.
    update_lidar_map()

    # Think/Act through FSM.
    if state == EXPLORE:
        handle_explore()
    elif state == TARGET_DETECTION:
        handle_target_detection()
    elif state == NAVIGATE_TO_TARGET:
        handle_navigate_to_target()
    elif state == RETRIEVAL_CONFIRMATION:
        handle_retrieval_confirmation()
    elif state == EXIT_NAVIGATION:
        handle_exit_navigation()
    elif state == FAILSAFE:
        handle_failsafe()
    elif state == MISSION_COMPLETE:
        handle_mission_complete()
    else:
        stop_robot()
        change_state(FAILSAFE)