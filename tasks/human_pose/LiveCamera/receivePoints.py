import socket
import json
import argparse
import threading

def handle_client(connection, address, camera_id):
    print(f"Accepted connection from {address} for Camera {camera_id}")

    try:
        while True:
            # Receive joint coordinates
            data = connection.recv(1024)
            if not data:
                break

            # Decode and process the received data (joint coordinates)
            received_joint_coordinates = json.loads(data.decode())
            print(f"Camera {camera_id} Joint Coordinates: {received_joint_coordinates}")

    except Exception as e:
        print(f"Error handling connection for Camera {camera_id}: {e}")

    finally:
        # Clean up the connection
        connection.close()

parser = argparse.ArgumentParser(description='Send joint coordinates over a network.')
parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of port numbers for communication with cameras.')
args = parser.parse_args()
receiver_ports = args.ports

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
    thread = threading.Thread(target=handle_client, args=(connection, address, i + 1))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Close all sockets
for sock in sockets:
    sock.close()
