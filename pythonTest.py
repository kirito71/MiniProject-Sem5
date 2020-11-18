gammaMax = 1e-8
gammaMin = 1e-10
gamma = gammaMin + ((gammaMax - gammaMin) / 255) * 135
print(gamma)