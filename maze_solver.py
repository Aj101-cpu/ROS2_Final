#!/usr/bin/env python3
"""EE3280 Lab 8 - Line Follower.

The original image-processing exercise is preserved: change `preview_image`
to publish the four OpenCV images one at a time.

STUDENT WORK: replace the THREE TODO expressions in the controller.

Motion is disabled during screenshot collection. Only run with
--ros-args -p enable_motion:=true after the stand/stop checks.
The software stop is best-effort, NOT an emergency stop.
"""
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped

class OpenCVSubscriber(Node):
    def __init__(self):
        super().__init__('opencv_subscriber')

        # Added for safe testing. Default is preview-only (no movement).
        self.declare_parameter('enable_motion', True)
        self.declare_parameter('kp', 0.25)
        self.declare_parameter('straight_speed', 0.25)
        self.declare_parameter('curve_speed', 0.15)
        self.declare_parameter('max_angular_speed', 0.25)
        self.enable_motion = bool(self.get_parameter('enable_motion').value)
        self.kp = float(self.get_parameter('kp').value)
        self.straight_speed = max(0.0, float(self.get_parameter('straight_speed').value))
        self.curve_speed = max(0.0, float(self.get_parameter('curve_speed').value))
        self.max_angular_speed = abs(float(self.get_parameter('max_angular_speed').value))

        # Original Lab 8 camera and output topics.
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, 'opencv_image', 10)
        self.publisher = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # Added safety and mild centroid smoothing.
        self.filtered_cx = None
        self.last_image_time = time.monotonic()
        self.watchdog_timer = self.create_timer(0.1, self.check_camera_timeout)
        self.get_logger().info(
            'Lab 8 started: enable_motion=' + str(self.enable_motion))

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        self.publisher.publish(cmd)

    def stop_robot(self):
        if self.enable_motion:
            self.publish_velocity(0.0, 0.0)

    def check_camera_timeout(self):
        # Stop if images stop arriving while the program is still running.
        if self.enable_motion and time.monotonic() - self.last_image_time > 0.4:
            self.stop_robot()

    def image_callback(self, msg):
        # 1. Convert compressed ROS image to OpenCV image via NumPy.
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return
        self.last_image_time = time.monotonic()

        # 2. Flip right side up.
        flipped_image = cv2.flip(cv_image, -1)
        height, width, _ = flipped_image.shape

        # 3. Crop top half to see floor and tape.
        fractional_height = int(height * 0.7)  # tune crop if needed
        cropped_image = flipped_image[fractional_height:height, 0:width]

        # 4-6. Gray, threshold, invert (original Lab 8 steps).
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 127, 255, 0)
        inverted_image = cv2.bitwise_not(thresh_image)

        # 7-9. Find the tape contour and its centroid.
        contours, hierarchy = cv2.findContours(
            inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Always build an output image (including frames with no contour).
        output_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
        linear_x = self.straight_speed  # default stop if no tape is detected
        angular_z = 0.0
        tape_found = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= 250.0:
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = int(M['m01'] / M['m00'])
                    tape_found = True

                    if self.filtered_cx is None:
                        self.filtered_cx = cx
                    else:
                        self.filtered_cx = 0.6 * self.filtered_cx + 0.4 * cx

                    cv2.circle(output_image,
                               (int(self.filtered_cx), cy),
                               5, (0, 0, 255), -1)
                    cv2.drawContours(output_image,
                                     [largest_contour], -1, (0, 255, 0), 2)

                    # ===== STUDENT WORK: THREE EXPRESSIONS =====
                    # TODO 1: Normalized signed error. Tape left -> positive.
                    error = (width / 2 - self.filtered_cx) / (width / 2)  # TODO 1: replace this expression

                    # Small deadband to reduce left-right oscillation.
                    if abs(error) < 0.06:
                        error = 0.0

                    # TODO 2: P controller (using self.kp and error).
                    angular_z = self.kp * error  # TODO 2: replace this expression
                    angular_z = max(-self.max_angular_speed,
                                    min(self.max_angular_speed, angular_z))

                    # TODO 3: Slow down when abs(error) >= 0.40.
                    # Use self.curve_speed and self.straight_speed.
                    linear_x = self.curve_speed if abs(error) >= 0.40 else self.straight_speed  # TODO 3: replace this expression
                    # ===== END STUDENT WORK =====

        if not tape_found:
            self.filtered_cx = None

        # ORIGINAL IMAGE EXERCISE: change the line below and take FOUR RQT
        # screenshots, one each for cropped_image, thresh_image,
        # inverted_image, and output_image.
        preview_image = output_image
        # Both grayscale and color previews work without changing encoding.
        encoding = 'bgr8' if preview_image.ndim == 3 else 'mono8'
        ros_image_message = self.bridge.cv2_to_imgmsg(
            preview_image, encoding=encoding)
        self.publisher_.publish(ros_image_message)

        # No motion while taking screenshots. Lost line -> zero command.
        if self.enable_motion:
            self.publish_velocity(linear_x, angular_z)

def main(args=None):
    rclpy.init(args=args)
    node = OpenCVSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort ROS stop during normal Ctrl+C shutdown.
        if node.enable_motion:
            for _ in range(5):
                node.stop_robot()
                time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
