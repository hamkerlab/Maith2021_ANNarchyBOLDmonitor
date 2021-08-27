import numpy as np
import pylab as plt


rB1=np.load('../dataRaw/Fig_Balloon_model_data_recordingsB_f_out_test_1.npy', allow_pickle=True).item()
rB2=np.load('../dataRaw/Fig_Balloon_model_data_recordingsB_f_out_test_2.npy', allow_pickle=True).item()

print(rB1.keys())

plt.figure(figsize=(16,9), dpi=500)
plt.subplot(211)
plt.title('f_out')
plt.plot(rB1['2;f_out'][:,0], color='k', label='0')
plt.plot(rB2['2;f_out'][:,0], color='r', label='20')
plt.subplot(212)
plt.title('v')
plt.plot(rB1['2;v'][:,0], color='k', label='0')
plt.plot(rB2['2;v'][:,0], color='r', label='20')
plt.legend()
plt.savefig('test.png')
