import sys
import time
import requests
from urllib.parse import urlparse
import socket
import csv
import ipaddress


def convert_to_binary(ip_adress):
    padded_binary = ""
    for part in ip_adress.split('.'):
        binary = bin(int(part)).replace("0b", '')
        padded_binary = padded_binary + binary.rjust(8, '0')
    return padded_binary

# function webping fetches the host address and url of pdf/image/video and calculates throughput for the file
# downloaded.

def webping(hostaddress, data_url):
    try:
        IP = socket.gethostbyname(hostaddress)
        bit_vector = convert_to_binary(str(IP))
        print("host and IP address ", hostaddress, bit_vector)
        # get function
        start = time.clock()
        myfile = requests.get(url=data_url, allow_redirects=True, timeout=2)
        print("request response", myfile)
        if myfile.status_code == 200:
            elapsed = (time.clock() - start)
            data_length = len(myfile.content)
            print("data", data_length)
            throughput = data_length / elapsed
            # throughput in kbytes
            throughput = throughput / 1000
            # throughput in megabytes
            throughput = throughput / 1000
            print("throughput", throughput)
            time.sleep(1)
            result = [IP, bit_vector, throughput,data_url, data_length, elapsed]
            return result
        else:
            return None

    except socket.gaierror as ge:
        print("OOPS!! socket Error.\n")
        print(str(ge))
        return None
    except requests.ConnectionError as e:
        print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")
        print(str(e))
        return None
    except requests.Timeout as e:
        print("OOPS!! Timeout Error")
        print(str(e))
        return None
    except requests.RequestException as e:
        print("OOPS!! General Error")
        print(str(e))
        return None
    except KeyboardInterrupt:
        print("Someone closed the program")
        return None




datafile = sys.argv[1]
outfile = sys.argv[2]
texturl = open(datafile, "r")

with open(outfile, 'w') as names:
    writer = csv.writer(names)
    header = ['Domain', 'IP-32bit', 'Throughput','url','datalength', 'elapsedtime']
    writer.writerow(header)

    for i in texturl:
        dataurl = i[:i.rindex(',')]
        print("url", dataurl)
        parsed = urlparse(dataurl)
        hostadd = parsed.netloc
        HOST = hostadd
        PORT = 80
        # fetches ip address of the host
        values = webping(HOST, dataurl)
        if values is not None:
            writer.writerow(values)
