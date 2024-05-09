# import torch
import numpy as np
import matplotlib.pyplot as plt


######################################
# Energy and time constants
######################################

# Activation
act_delay = np.array(7.72e-9, dtype='float128')
act_power = np.array(8.98e-3, dtype='float128')
act_1bit_e = act_delay * act_power
# Activation new
act_delay = np.array(1 / 254e6, dtype='float128')
act_power = np.array(0.81e-6, dtype='float128')
act_1bit_e = act_delay * act_power

# RRAM
v_RRAM = 0.3  # Mean inference voltage of RRAM, in V
# v_RRAM = 0.3  # Mean inference voltage of RRAM, in V
t_RRAM = np.array(1e-8, dtype='float128')  # Mean inference voltage pulse width, in s
# g_avg_rram = 14.2473e-6
# g_avg_rram = 13.00e-6 / 2 # half sparse
g_avg_rram = 33.717e-6 / 2 # half sparse
e_avg_rram = 3.3717e-15 / 2 #  = v_RRAM * v_RRAM * g_avg_rram * t_RRAM

# ADC
# erg_TIAADC_x1 = 1.26e-3 * t_RRAM  # One time TIA ADC operation energy, in J
erg_ADC_x1 = np.array(1.13e-3 * t_RRAM, dtype='float128')
# 2.5mJ * 20e-6

# DAC
# erg_DAC_x1 = np.array(26.4e-6 * 2.12e-6, dtype='float128')  # One time DAC operation energy, in J 26.4uw * 2.12us
erg_DAC_x1 = np.array(227e-15, dtype='float128')  # One time DAC operation energy, in J
# from "An 8-Bit Single-Ended Ultra-Low-Power SAR ADC With a Novel DAC Switching Method and a Counter-Based Digital Control Circuitry"


# Decoder
# erg_decoder = np.array(0.08e-12 / 32, dtype='float128')
erg_decoder = np.array(0.42e-15, dtype='float128')
# MUX
erg_MUX = np.array(14.06e-15, dtype='float128') # 0.9pJ
# WL & BL driver
erg_driver = np.array(0.18e-15, dtype='float128')

# -------------------------
# Digital energy estimation
# -------------------------
# gtx1080
t_gtx1080 = np.array(8.9e12, dtype='float128')  # Throughput (FP operations/s)
p_gtx1080 = 180     # TDP (W)
erg_1080 = p_gtx1080 / t_gtx1080  # Energy per FP operation (J)
# rtx3090
t_rtx3090 = np.array(284e12, dtype='float128')  # Throughput (FP operations/s)
p_rtx3090 = 350 # TDP (W)
erg_rtx3090 = p_rtx3090 / t_rtx3090  # Energy per FP operation (J)
# A100
t_a100_int8 = np.array(624e12, dtype='float128') #
p_a100 = 300  # TDP (W)
erg_a100 = p_a100 / t_a100_int8  # Energy per FP operation (J)
# # rtx4090
# t_rtx3090 = np.array(82.58e12, dtype='float128')  # Throughput (FP operations/s)
# p_rtx3090 = 450 # TDP (W)
# erg_rtx3090 = p_rtx3090 / t_rtx3090  # Energy per FP operation (J)

erg_gpu_dict = {'1080': erg_1080, '3090': erg_rtx3090, 'a100': erg_a100}





def analog_system_energy(gpu='3090'):

    gpu_e = erg_gpu_dict[gpu]

    layer_op=[40278,11808,11232,10944,10560,10560,10560,10560,10560,10560,10560,288]
    layer_rram = [37926,11424,11040,10848,10512,10512,10512,10512,10512,10512,10512,240]
    layer_digital = [2352,384,192,96,48,48,48,48,48,48,48,48]

    ternary = [0,8,21,9957,9,0,0,0,0,1,0,4]
    ternary_01 = [0,15,13,9967,2,2,0,0,0,0,0,1]
    ternary_05 = [0,0,1,9999,0,0,0,0,0,0,0,0]



    'layer1'
    # gpu energy
    layer1_e_gpu = gpu_e * layer_op[0]
    # mixedsystem rram energy
    # DAC
    conv1_e_dac = erg_DAC_x1 * 28 * 28 * 1
    layer1_1_e_dac = erg_DAC_x1 * 7 * 7 * 3
    layer1_2_e_dac = erg_DAC_x1 * 7 * 7 * 3
    # VMM
    conv1_layer1_e_rram = e_avg_rram * (layer_rram[0] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    conv1_e_adc = erg_ADC_x1 * 14 * 14 * 3
    layer1_1_e_adc = erg_ADC_x1 * 7 * 7 * 3
    layer1_2_e_adc = erg_ADC_x1 * 7 * 7 * 3
    e_cim_adc_layer1 = conv1_e_adc+ layer1_1_e_adc+ layer1_2_e_adc
    # mixedsystem gpu energy
    layer1_e_mixgpu = gpu_e * layer_digital[0]
    # mixedsystem energy
    # layer1_e_mix = conv1_e_dac + layer1_1_e_dac + layer1_2_e_dac + conv1_layer1_e_rram + conv1_e_adc + layer1_1_e_adc + \
    # layer1_2_e_adc + layer1_e_mixgpu
    #non-dac
    layer1_e_mix = conv1_layer1_e_rram + conv1_e_adc + layer1_1_e_adc + \
    layer1_2_e_adc + layer1_e_mixgpu
    
    print(f'layer1_e_mix:{layer1_e_mix}')
    print(f'conv1_layer1_e_rram:{conv1_layer1_e_rram}')
    print(f'layer1_e_mixgpu:{layer1_e_mixgpu}')
    print(f' e_cim_adc_layer1:{ e_cim_adc_layer1}')

    'layer2'
    # gpu energy
    layer2_e_gpu = gpu_e * layer_op[1]
    # mixedsystem rram energy
    # DAC
    conv2_e_dac = erg_DAC_x1 * 7 * 7 * 3
    layer2_1_e_dac = erg_DAC_x1 * 4 * 4 * 6
    layer2_2_e_dac = erg_DAC_x1 * 4 * 4 * 6
    # VMM
    conv2_layer2_e_rram = e_avg_rram * (layer_rram[1] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    conv2_e_adc = erg_ADC_x1 * 4 * 4 * 6
    layer2_1_e_adc = erg_ADC_x1 * 4 * 4 * 6
    layer2_2_e_adc = erg_ADC_x1 * 4 * 4 * 6
    e_cim_adc_layer2 = conv2_e_adc+ layer2_1_e_adc+ layer2_2_e_adc
    # mixedsystem gpu energy
    layer2_e_mixgpu = gpu_e * layer_digital[1]
    # mixedsystem energy
    # layer2_e_mix = conv2_e_dac + layer2_1_e_dac + layer2_2_e_dac + conv2_layer2_e_rram + conv2_e_adc + layer2_1_e_adc + \
    # layer2_2_e_adc + layer2_e_mixgpu
    #non-dac
    layer2_e_mix = conv2_layer2_e_rram + conv2_e_adc + layer2_1_e_adc + \
    layer2_2_e_adc + layer2_e_mixgpu

    print(f'layer2_e_mix:{layer2_e_mix}')
    print(f'conv2_layer2_e_rram:{conv2_layer2_e_rram}')
    print(f'layer2_e_mixgpu:{layer2_e_mixgpu}')
    print(f' e_cim_adc_layer2:{ e_cim_adc_layer2}')

    'layer3'
    # gpu energy
    layer3_e_gpu = gpu_e * layer_op[2]
    # mixedsystem rram energy
    # DAC
    conv3_e_dac = erg_DAC_x1 * 4 * 4 * 6
    layer3_1_e_dac = erg_DAC_x1 * 2 * 2 * 12
    layer3_2_e_dac = erg_DAC_x1 * 2 * 2 * 12
    # VMM
    conv3_layer3_e_rram = e_avg_rram * (layer_rram[2] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    conv3_e_adc = erg_ADC_x1 * 2 * 2 * 12
    layer3_1_e_adc = erg_ADC_x1 * 2 * 2 * 12
    layer3_2_e_adc = erg_ADC_x1 * 2 * 2 * 12
    e_cim_adc_layer3 = conv3_e_adc + layer3_1_e_adc + layer3_2_e_adc
    # mixedsystem gpu energy
    layer3_e_mixgpu = gpu_e * layer_digital[2]
    # mixedsystem energy
    # layer3_e_mix = conv3_e_dac + layer3_1_e_dac + layer3_2_e_dac + conv3_layer3_e_rram + conv3_e_adc + layer3_1_e_adc + \
    # layer3_2_e_adc + layer3_e_mixgpu
    #non-dac
    layer3_e_mix = conv3_layer3_e_rram + conv3_e_adc + layer3_1_e_adc + \
    layer3_2_e_adc + layer3_e_mixgpu

    print(f'layer3_e_mix:{layer3_e_mix}')
    print(f'conv3_layer3_e_rram:{conv3_layer3_e_rram}')
    print(f'layer3_e_mixgpu:{layer3_e_mixgpu}')
    print(f' e_cim_adc_layer3:{ e_cim_adc_layer3}')

    'layer4'
    # gpu energy
    layer4_e_gpu = gpu_e * layer_op[3]
    # mixedsystem rram energy
    # DAC
    conv4_e_dac = erg_DAC_x1 * 2 * 2 * 12
    layer4_1_e_dac = erg_DAC_x1 * 1 * 1 * 24
    layer4_2_e_dac = erg_DAC_x1 * 1 * 1 * 24
    # VMM
    conv4_layer4_e_rram = e_avg_rram * (layer_rram[3] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    conv4_e_adc = erg_ADC_x1 * 1 * 1 * 24
    layer4_1_e_adc = erg_ADC_x1 * 1 * 1 * 24
    layer4_2_e_adc = erg_ADC_x1 * 1 * 1 * 24
    e_cim_adc_layer4 = conv4_e_adc+ layer4_1_e_adc+ layer4_2_e_adc
    # mixedsystem gpu energy
    layer4_e_mixgpu = gpu_e * layer_digital[3]
    # mixedsystem energy
    # layer4_e_mix = conv4_e_dac + layer4_1_e_dac + layer4_2_e_dac + conv4_layer4_e_rram + conv4_e_adc + layer4_1_e_adc + \
    # layer4_2_e_adc + layer4_e_mixgpu
    #non-dac
    layer4_e_mix = conv4_layer4_e_rram + conv4_e_adc + layer4_1_e_adc + \
    layer4_2_e_adc + layer4_e_mixgpu

    print(f'layer4_e_mix:{layer4_e_mix}')
    print(f'conv4_layer4_e_rram:{conv4_layer4_e_rram}')
    print(f'layer4_e_mixgpu:{layer4_e_mixgpu}')
    print(f' e_cim_adc_layer4:{ e_cim_adc_layer4}')

    'layer5'
    # gpu energy
    layer5_e_gpu = gpu_e * layer_op[4]
    # mixedsystem rram energy
    # DAC
    # conv4_e_dac = erg_DAC_x1 * 2 * 2 * 12
    layer5_1_e_dac = erg_DAC_x1 * 1 * 1 * 24
    layer5_2_e_dac = erg_DAC_x1 * 1 * 1 * 24
    # VMM
    layer5_e_rram = e_avg_rram * (layer_rram[4] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    # conv4_e_adc = erg_ADC_x1 * 1 * 1 * 24
    layer5_1_e_adc = erg_ADC_x1 * 1 * 1 * 24
    layer5_2_e_adc = erg_ADC_x1 * 1 * 1 * 24
    e_cim_adc_layer5 = layer5_1_e_adc + layer5_2_e_adc
    # mixedsystem gpu energy
    layer5_e_mixgpu = gpu_e * layer_digital[4]
    # mixedsystem energy
    # layer5_e_mix = layer5_1_e_dac + layer5_2_e_dac + layer5_e_rram + layer5_1_e_adc + \
    # layer5_2_e_adc + layer5_e_mixgpu
    #non_dac
    layer5_e_mix = layer5_e_rram + layer5_1_e_adc + \
    layer5_2_e_adc + layer5_e_mixgpu

    print(f'layer5_e_mix:{layer5_e_mix}')
    print(f'conv5_layer5_e_rram:{layer5_e_rram}')
    print(f'layer5_e_mixgpu:{layer5_e_mixgpu}')
    print(f' e_cim_adc_layer5:{ e_cim_adc_layer5}')

    'layer12'
    # gpu energy
    layer12_e_gpu = gpu_e * layer_op[11]
    # mixedsystem rram energy
    # DAC
    # conv4_e_dac = erg_DAC_x1 * 2 * 2 * 12
    layer12_e_dac = erg_DAC_x1 * 1 * 1 * 24
    # VMM
    layer12_e_rram = e_avg_rram * (layer_rram[11] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    # conv4_e_adc = erg_ADC_x1 * 1 * 1 * 24
    layer12_e_adc = erg_ADC_x1 * 1 * 1 * 10
    e_cim_adc_layer12 = layer12_e_adc
    # mixedsystem gpu energy
    layer12_e_mixgpu = gpu_e * layer_digital[11]
    # mixedsystem energy
    layer12_e_mix = layer12_e_dac + layer12_e_rram + layer12_e_adc + \
    layer12_e_mixgpu

    print(f'layer12_e_mix:{layer12_e_mix}')
    print(f'conv12_layer12_e_rram:{layer12_e_rram}')
    print(f'layer12_e_mixgpu:{layer12_e_mixgpu}')
    print(f' e_cim_adc_layer12:{ e_cim_adc_layer12}')
    


    #Total energy
    e_gpu_system = layer1_e_gpu + layer2_e_gpu + layer3_e_gpu + layer4_e_gpu + layer5_e_gpu * 7 + layer12_e_gpu
    e_gpu_system = 100*e_gpu_system
    e_gpu_early_exit_system = 100 * layer1_e_gpu + (100) * layer2_e_gpu + (100) * layer3_e_gpu + (100) * layer4_e_gpu + \
                                           (3) * layer5_e_gpu + 0 * layer12_e_gpu
    e_mixed_system = layer1_e_mix + layer2_e_mix + layer3_e_mix + layer4_e_mix + layer5_e_mix * 7 + layer12_e_mix
    e_mixed_system = 100*e_mixed_system

    #cosin_energy
    cos_e_gpu_system = [10,10,10,10,10,10,10,10,10,10,10,10]
    cos_e_rram_system = [30,60,120,240,240,240,240,240,240,240,240,240]

    #early_exit
    num =10000
    #ternary
    e_mix_early_exit_ternary_conv_system = 100 * layer1_e_mix + 100 * layer2_e_mix + (100) * layer3_e_mix + (100) * layer4_e_mix + \
                                           (1+1+1) * layer5_e_mix + 0 * layer12_e_mix
    cos_ternary_rram_num = 100*cos_e_rram_system[0]+100*cos_e_rram_system[1]+(100)*cos_e_rram_system[2]+(100)*cos_e_rram_system[3]+\
                          (1+1+1)*cos_e_rram_system[4]+ 0*cos_e_rram_system[11]
    cos_ternary_gpu_num = 100+100+(100)+(100)+(1+1+1)
    e_cos_ternary_gpu = cos_ternary_gpu_num * 10 * gpu_e + cos_ternary_gpu_num * 10 * gpu_e + cos_ternary_rram_num * gpu_e #shift(+1) + 除法 + 范数（norm）
    e_cos_ternary_rram = e_avg_rram * (cos_ternary_rram_num / 2) * 2 #cosine multiply

    e_cam =  e_cos_ternary_gpu +  e_cos_ternary_rram
    e_cam_rram = e_cos_ternary_rram
    e_cam_gpu = e_cos_ternary_gpu
    e_cam_adc = erg_ADC_x1 * cos_ternary_gpu_num * 10

    e_cim =  e_mix_early_exit_ternary_conv_system
    e_cim_rram =  100 * conv1_layer1_e_rram + 100 * conv2_layer2_e_rram + (100) * conv3_layer3_e_rram + (100) * conv4_layer4_e_rram+ \
                                           (1+1+1) * layer5_e_rram + 0 * layer12_e_rram
    e_cim_gpu = 100 * layer1_e_mixgpu + 100 * layer2_e_mixgpu + (100) * layer3_e_mixgpu + (100) * layer4_e_mixgpu+ \
                                           (1+1+1) * layer5_e_mixgpu + 0 * layer12_e_mixgpu
    e_cim_adc = 100 * e_cim_adc_layer1 + 100 * e_cim_adc_layer2 + (100) * e_cim_adc_layer3 + (100) * e_cim_adc_layer4+ \
                                           (1+1+1) * e_cim_adc_layer5 + 0 * e_cim_adc_layer12
    print(f'e_cam:{e_cam},e_cam_rram:{e_cam_rram},e_cam_gpu:{e_cam_gpu},e_cam_adc:{e_cam_adc}')
    print(f'e_cim:{e_cim},e_cim_rram:{e_cim_rram},e_cim_gpu:{e_cim_gpu},e_cim_adc:{e_cim_adc}')



    e_mix_early_exit_ternary_system = e_mix_early_exit_ternary_conv_system + e_cos_ternary_gpu + e_cos_ternary_rram + e_cam_adc

    #rram_system
    e_rram_early_exit_system = 10000 * conv1_layer1_e_rram + 10000 * conv2_layer2_e_rram + (10000-416) * conv3_layer3_e_rram + (10000-416-9554) * conv4_layer4_e_rram+ \
                                           (30+27+25+10+9+9+9) * layer5_e_rram + 9 * layer12_e_rram +  e_cos_ternary_rram 
    e_cos_ternary_rram_total = e_avg_rram * (10000*cos_e_rram_system[0]+10000*cos_e_rram_system[1]+(10000)*cos_e_rram_system[2]+(10000)*cos_e_rram_system[3]+\
                          (10000*7)*cos_e_rram_system[4]+ 10000*cos_e_rram_system[11])
    e_rram_early_total_system = 10000 * conv1_layer1_e_rram + 10000 * conv2_layer2_e_rram + 10000 * conv3_layer3_e_rram + (10000) * conv4_layer4_e_rram+ \
                                           (10000*7) * layer5_e_rram + 10000 * layer12_e_rram + e_cos_ternary_rram_total


    # print(f'gpu_system:{e_gpu_system},mix_system:{e_mixed_system}, e_gpu_early_exit_system:{e_gpu_early_exit_system},mix_early_exit_ternary_system:{e_mix_early_exit_ternary_system},'
    #       f'mix_early_exit_ternary_01_system:{e_mix_early_exit_ternary_01_system},mix_early_exit_ternary_05_system:{e_mix_early_exit_ternary_05_system},e_gpu_early_exit_system:{e_gpu_early_exit_system}')

    print(f'gpu_system:{e_gpu_system},mix_system:{e_mixed_system}, mix_early_exit_ternary_system:{e_mix_early_exit_ternary_system},e_gpu_early_exit_system:{e_gpu_early_exit_system}\
    ,e_rram_early_exit_system:{e_rram_early_exit_system},e_rram_early_total_system:{e_rram_early_total_system}')


if __name__ == '__main__':
    analog_system_energy(gpu='3090')
