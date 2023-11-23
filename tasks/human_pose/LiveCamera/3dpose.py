import socket
import json
import argparse
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_pose(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    # Plot 3D points
    ax.scatter(x, y, z, c='r', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def triangulate(joints_cam1, joints_cam2, P1, P2):
    # Perform triangulation to estimate 3D coordinates
    homogeneous_coords_cam1 = np.hstack((np.array(joints_cam1), np.ones((len(joints_cam1), 1))))
    homogeneous_coords_cam2 = np.hstack((np.array(joints_cam2), np.ones((len(joints_cam2), 1))))

    # Use OpenCV's triangulatePoints to perform triangulation
    points_4d = cv2.triangulatePoints(P1, P2, homogeneous_coords_cam1.T, homogeneous_coords_cam2.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).reshape(-1, 3)

    return points_3d

def handle_client(connection, address, camera_id, calibration_matrices):
    print(f"Accepted connection from {address} for Camera {camera_id}")

    try:
        buffer = b""
        joints_cam1 = None
        joints_cam2 = None

        while True:
            # Receive joint coordinates
            data = connection.recv(1024)
            if not data:
                break

            # Accumulate data in the buffer
            buffer += data

            while b"]][[" in buffer:
                # Split the buffer into separate JSON objects
                start_index = buffer.find(b"]][[") + 2
                json_obj = buffer[:start_index]
                buffer = buffer[start_index:]

                try:
                    # Attempt to decode a JSON object
                    received_joint_coordinates = json.loads(json_obj.decode())
                    print(f"Camera {camera_id} Joint Coordinates: {received_joint_coordinates}")

                    if camera_id == 1:
                        joints_cam1 = received_joint_coordinates
                    elif camera_id == 2:
                        joints_cam2 = received_joint_coordinates

                    if joints_cam1 and joints_cam2:
                        # Perform triangulation when coordinates from both cameras are available
                        P1, P2 = calibration_matrices[camera_id - 1]
                        # Ensure that P1 and P2 are the correct camera matrices obtained during calibration
                        P1, P2 = np.array(P1), np.array(P2)

                        # Perform triangulation to get 3D coordinates
                        points_3d = triangulate(joints_cam1, joints_cam2, P1, P2)
                        print(f"3D Coordinates: {points_3d}")

                        # Visualize the 3D pose
                        visualize_3d_pose(points_3d)

                except json.JSONDecodeError:
                    # Continue accumulating data if a complete JSON object is not yet received
                    continue

    except Exception as e:
        print(f"Error handling connection for Camera {camera_id}: {e}")

    finally:
        # Clean up the connection
        connection.close()

def main():
    parser = argparse.ArgumentParser(description='Send joint coordinates over a network.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of port numbers for communication with cameras.')
    args = parser.parse_args()
    receiver_ports = args.ports

    # Sample intrinsic parameters obtained during camera calibration
    fx = 1000.0  # Focal length in pixels (along x-axis)
    fy = 1000.0  # Focal length in pixels (along y-axis)
    cx = 640.0   # Principal point x-coordinate in pixels
    cy = 480.0   # Principal point y-coordinate in pixels

    # Define the camera matrix (intrinsic matrix)
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    # Assuming no distortion for simplicity
    dist_coeffs = np.zeros((4, 1))

    # Create the projection matrix P1 for camera 1
    R1 = np.eye(3)  # Identity rotation matrix
    T1 = np.zeros((3, 1))  # Translation vector

    P1 = np.hstack((K, np.zeros((3, 1))))

    # Create the projection matrix P2 for camera 2 (assuming a baseline translation)
    baseline_translation = 0.1  # Example baseline translation in meters
    T2 = np.array([[baseline_translation], [0], [0]])

    # Rotation matrix for camera 2 (you may need to adjust this based on your setup)
    # For simplicity, we assume the cameras are parallel, so R2 is the same as R1
    R2 = R1

    P2 = np.hstack((K, -R2 @ T2))

    # Sample values for P1 and P2 matrices
    sample_P1 = P1.tolist()
    sample_P2 = P2.tolist()

    # Now you can use these sample values in your main code
    calibration_matrices = [sample_P1, sample_P2]

    # Create a socket for each camera
    sockets = []
    for i, port in enumerate(receiver_ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', port))
        sock.listen(1)
        sockets.append(sock)

    # Create a thread for each camera
    threads = []
    for i, sock in enumerate(sockets):
        connection, address = sock.accept()
        thread = threading.Thread(target=handle_client, args=(connection, address, i + 1, calibration_matrices))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Close all sockets
    for sock in sockets:
        sock.close()

if __name__ == "__main__":
    main()
