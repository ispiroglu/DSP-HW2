import matplotlib.pyplot as plt # Plotting only

def sig_square(x):
  return 0 if x < 3 or x > 5 else 2

def sig_triag(x):
  return 0 if x < 0 or x > 2 else x

# First signal (square pulse)
sig1 = [sig_square(x/100) for x in range(1000)]

# Seconds signal (triangle pulse)
sig2 = [sig_triag(x/100) for x in range(200)]

conv = (len(sig1) - len(sig2)) * [0]



for l in range(len(conv)):
  for i in range(len(sig2)):
    conv[l] += sig1[l-i+len(sig2)] * sig2[i]

  conv[l] /= len(sig2) # Normalize

plt.plot(conv)
plt.show()
