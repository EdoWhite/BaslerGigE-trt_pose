import socket
import json
import argparse
import threading

def handle_client(connection, address, camera_id):
    print(f"Accepted connection from {address} for Camera {camera_id}")

    try:
        buffer = b""
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

                except json.JSONDecodeError:
                    # Continue accumulating data if a complete JSON object is not yet received
                    continue
    except Exception as e:
        print(f"Error handling connection for Camera {camera_id}: {e}")

    finally:
        # Clean up the connection
        connection.close()


parser = argparse.ArgumentParser(description='Send joint coordinates over a network.')
parser.add_argument('--ip', type=int, required=True, help='IP of th receiver machine')
parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of port numbers for communication with cameras.')
args = parser.parse_args()
receiver_ports = args.ports
receiver_ip = args.ip

# Create a socket for each camera
sockets = []
for i, port in enumerate(receiver_ports):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((receiver_ip, port))
    sockets.append(sock)

# Create a thread for each camera
threads = []
for i, sock in enumerate(sockets):
    #connection, address = sock.recvfrom(1024)
    thread = threading.Thread(target=handle_client, args=(sock, None, i + 1))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Close all sockets
for sock in sockets:
    sock.close()
