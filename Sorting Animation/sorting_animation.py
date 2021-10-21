import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrackedArray: # 
    def __init__(self, arr, kind="not full"):
        self.arr = np.copy(arr)
        self.kind = kind
        self.reset()

    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []

    def track(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        if self.kind == "full":
            self.full_copies.append(np.copy(self.arr))

    def GetActivity(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])

    def __getitem__(self, key):
        self.track(key, 'get')
        return self.arr.__getitem__(key)
    
    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.track(key, 'set')
    
    def __len__(self):
        return self.arr.__len__()

def update(frame):
    steps_counter.set_text(f'Steps = {frame}')
    for (rectangle, height) in zip(container.patches, arr.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color('#9400D3')
    
    idx, op = arr.GetActivity(frame)
    if op == "get":
        container.patches[idx].set_color("#FFA500")
    elif op == "set":
        container.patches[idx].set_color("#FFD700")
    return (*container, steps_counter)

plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams['font.size'] = 16
FPS = 60.0
N = 50
arr = np.round(np.linspace(0,1000,N),0)
np.random.seed(0)
np.random.shuffle(arr)
arr = TrackedArray(arr, 'full')

################################################
################# Bubble Sort ##################
################################################
def bubble_sort(arr):
   for iter_num in range(len(arr)-1,0,-1):
      for idx in range(iter_num):
         if arr[idx]>arr[idx+1]:
            arr[idx], arr[idx+1] = arr[idx+1], arr[idx]

################################################
################# Quick Sort ###################
################################################
def quick_sort(arr, lo, hi):
    if lo < hi:
        p = partition(arr, lo, hi)
        quick_sort(arr, lo, p-1)
        quick_sort(arr, p+1, hi)

def partition(arr, lo, hi):
    pivot, i = arr[hi], lo
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i+=1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i

################################################
################# Merge Sort ###################
################################################
def merge(arr, l, m, r):
	n1 = m-l+1
	n2 = r-m
	L, R = [0]*(n1), [0]*(n2) 
	for i in range(0, n1):
		L[i] = arr[l+i]
	for j in range(0, n2):
		R[j] = arr[m+1+j]
	i, j, k = 0, 0, l	 

	while i < n1 and j < n2:
		if L[i] <= R[j]:
			arr[k] = L[i]
			i += 1
		else:
			arr[k] = R[j]
			j += 1
		k += 1
	while i < n1:
		arr[k] = L[i]
		i += 1
		k += 1

	while j < n2:
		arr[k] = R[j]
		j += 1
		k += 1

def merge_sort(arr, l, r):
	if l < r:
		m = l+(r-l)//2
		merge_sort(arr, l, m)
		merge_sort(arr, m+1, r)
		merge(arr, l, m, r)

################################################
############### Inserion Sort ##################
################################################
def instertion_sort(arr):
    i = 1
    while i<len(arr):
        j=i
        while (j>0 and arr[j-1] > arr[j]):
            arr[j-1], arr[j] = arr[j], arr[j-1]
            j-=1
        i+=1
    return arr

################################################
################# Shell Sort ###################
################################################
def shell_sort(array, n):
    interval = n // 2
    while interval > 0:
        for i in range(interval, n):
            temp = array[i]
            j = i
            while j >= interval and array[j - interval] > temp:
                array[j] = array[j - interval]
                j -= interval
            array[j] = temp
        interval //= 2
################################################

####################
######Sorting#######
####################
########################## Names of Sorting Algoritms #########################
# sorter = 'Bubble Sort'
# sorter = 'Quick Sort'
# sorter = 'Merge Sort'
# sorter = 'Instertion Sort'
# sorter = 'Shell Sort'

sorting_algoritms = {1:'Bubble Sort', 2:'Quick Sort', 3:'Merge Sort', 4:'Instertion Sort',5:'Shell Sort'}
print('\tSelect sorting Algoritm')
sa = input(f'1 : {sorting_algoritms[1]}\n2 : {sorting_algoritms[2]}\n3 : {sorting_algoritms[3]}\n4 : {sorting_algoritms[4]}\n5 : {sorting_algoritms[5]}\n:')
t0=time.perf_counter()
if sa == '1':
    sorter = sorting_algoritms[1]   
    bubble_sort(arr)
elif sa == '2':
    sorter = sorting_algoritms[2]
    quick_sort(arr, 0, len(arr)-1)
elif sa == '3':
    sorter = sorting_algoritms[3]
    merge_sort(arr, 0, len(arr)-1)
elif sa == '4': 
    sorter = sorting_algoritms[4]
    instertion_sort(arr)
elif sa == '5': 
    sorter = sorting_algoritms[5]
    shell_sort(arr, len(arr))
else:
    print('Bad input! Pick number from 1 to 5')
    quit()
########################## Sorting Algoritms #################################
# bubble_sort(arr)
# quick_sort(arr, 0, len(arr)-1)
# merge_sort(arr, 0, len(arr)-1)
# instertion_sort(arr)
# shell_sort(arr, len(arr))
dt = time.perf_counter() - t0
print(f'\t--{sorter}--')
print(f'\tTime: {dt*1E3:.1f} ms')

fig, ax = plt.subplots()
container = ax.bar(np.arange(0, len(arr), 1), arr, align='edge', width=0.8)
ax.set_xlim(0,N)
ax.set(xlabel='Index', ylabel='Value', title=f'{sorter}')
steps_counter = ax.text(0,1000, "")
ani = FuncAnimation(fig, update, frames=range(len(arr.full_copies)), blit=True, interval=1000./FPS, repeat=False)
plt.show()

