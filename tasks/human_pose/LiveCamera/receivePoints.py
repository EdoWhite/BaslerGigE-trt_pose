import socket
import json

# Create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
receiver_address = ('', 12345)  # Use an empty string for the address to accept connections on any available network interface
sock.bind(receiver_address)

# Listen for incoming connections
sock.listen(1)

print("Waiting for a connection...")

# Accept a connection
connection, client_address = sock.accept()

try:
    while True:
        # Receive joint coordinates
        data = connection.recv(1024)
        if not data:
            break

        # Decode and process the received data (joint coordinates)
        received_joint_coordinates = json.loads(data.decode())
        print(received_joint_coordinates)

finally:
    # Clean up the connection
    connection.close()
    sock.close()
