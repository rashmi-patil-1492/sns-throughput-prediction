import csv
import socket
import time
import socket
import ipaddress

stored_time = []
total_throughput = 0

# for i in range(1, n):
def webping(hostaddr):
	try:
		IP = socket.gethostbyname(hostaddr)
		print(IP) #print the IP address

			#convert ip string in to 32 bit vector
		bit_vector = (bin(int(ipaddress.IPv4Address(IP))).replace("0b", ""))
		print(bit_vector)

		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.connect((hostaddr, PORT))

			#print("bye")

			s.send(b"GET / HTTP/1.1\r\n\r\n")
			time.sleep(10)
			while True:
				data = s.recv(1024)
				print("recieved from the server", data)
				length_data = len(data)
				print("data length is", length_data)
				break
			s.close()

		# elapsed = (time.clock() - start)* 1000
		# throughput = length_data/elapsed
		# total_throughput += throughput
		# print("throughput is ", (total_throughput/n))

		print("elpased time in sec", elapsed)
		# print("total bytes", length_data * n)

		#total_time = sum(stored_time)
		#data_len = len(stored_time)
		# throughput = length_data * n/ elapsed
		print("throughput is ", throughput)

		result =[IP, bit_vector, throughput]
		return result
	except ConnectionResetError as connectionerror:
		print("connection establishment error")
	# except Traceback as tracebackerror:
	# 	print("traceback error")

with open('sampleurl.csv') as urlcsv:
	url = csv.reader(urlcsv, delimiter = ',')
	with open('domainnames.csv', 'w') as names:
		writer = csv.writer(names)
		header = ['Domain','IP-32bit','Throughput']
		writer.writerow(header)

		for name in url:
			HOST = name[0]
			PORT = 80
			# fetches ip address of the host
			values = webping(HOST)
			if values != None:
				print("this host didn't respond", HOST)
				continue

			writer.writerow(values)
