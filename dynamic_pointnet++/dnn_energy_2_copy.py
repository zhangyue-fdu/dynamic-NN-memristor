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

erg_gpu_dict = {'1080': erg_1080, '3090': erg_rtx3090, 'a100': erg_a100}





def analog_system_energy(gpu='3090'):

    gpu_e = erg_gpu_dict[gpu]

    layer_op=[221249536,355467264,355467264,355467264,1113587712,691011584,691011584,93290496,660992]
    layer_rram = [204472320,338690048,338690048,338690048,1080033280,674234368,674234368,92372992,657920]
    layer_digital = [16777216,16777216,16777216,16777216,33554432,16777216,16777216,917504,3072]

    ternary = [0,8,21,9957,9,0,0,0,0,1,0,4]
    ternary_01 = [0,15,13,9967,2,2,0,0,0,0,0,1]
    ternary_05 = [0,0,1,9999,0,0,0,0,0,0,0,0]

#PointNet++

    'layer1'
    # gpu energy
    layer1_e_gpu = gpu_e * layer_op[0]
    # mixedsystem rram energy
    # DAC
    conv1_e_dac = erg_DAC_x1 * 28 * 28 * 1
    layer1_1_e_dac = erg_DAC_x1 * 7 * 7 * 3
    layer1_2_e_dac = erg_DAC_x1 * 7 * 7 * 3
    # VMM
    layer1_e_rram = e_avg_rram * (layer_rram[0] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer1_e_adc = erg_ADC_x1 * 128 * 32 * 512
    # layer1_1_e_adc = erg_ADC_x1 * 7 * 7 * 3
    # layer1_2_e_adc = erg_ADC_x1 * 7 * 7 * 3
    # mixedsystem gpu energy
    layer1_e_mixgpu = gpu_e * layer_digital[0]
    # mixedsystem energy
    # layer1_e_mix = conv1_e_dac + layer1_1_e_dac + layer1_2_e_dac + conv1_layer1_e_rram + conv1_e_adc + layer1_1_e_adc + \
    # layer1_2_e_adc + layer1_e_mixgpu
    #non-dac
    layer1_e_mix = layer1_e_rram + + layer1_e_adc + layer1_e_mixgpu

    'layer2'
    # gpu energy
    layer2_e_gpu = gpu_e * layer_op[1]
    # mixedsystem rram energy
    # VMM
    layer2_e_rram = e_avg_rram * (layer_rram[1] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer2_e_adc = erg_ADC_x1 * 128 * 32 * 512
    # mixedsystem gpu energy
    layer2_e_mixgpu = gpu_e * layer_digital[1]
    # mixedsystem energy
    #non-dac
    layer2_e_mix = layer2_e_rram + + layer2_e_adc + layer2_e_mixgpu

    'layer3'
    # gpu energy
    layer3_e_gpu = gpu_e * layer_op[2]
    # mixedsystem rram energy
    # VMM
    layer3_e_rram = e_avg_rram * (layer_rram[2] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer3_e_adc = erg_ADC_x1 * 128 * 32 * 512
    # mixedsystem gpu energy
    layer3_e_mixgpu = gpu_e * layer_digital[2]
    # mixedsystem energy
    #non-dac
    layer3_e_mix = layer3_e_rram + + layer3_e_adc + layer3_e_mixgpu

    'layer4'
    # gpu energy
    layer4_e_gpu = gpu_e * layer_op[3]
    # mixedsystem rram energy
    # VMM
    layer4_e_rram = e_avg_rram * (layer_rram[3] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer4_e_adc = erg_ADC_x1 * 128 * 32 * 512
    # mixedsystem gpu energy
    layer4_e_mixgpu = gpu_e * layer_digital[3]
    # mixedsystem energy
    #non-dac
    layer4_e_mix = layer4_e_rram + + layer4_e_adc + layer4_e_mixgpu

    'layer5'
    # gpu energy
    layer5_e_gpu = gpu_e * layer_op[4]
    # mixedsystem rram energy
    # VMM
    layer5_e_rram = e_avg_rram * (layer_rram[4] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer5_e_adc = erg_ADC_x1 * 256 * 32 * 512
    # mixedsystem gpu energy
    layer5_e_mixgpu = gpu_e * layer_digital[4]
    # mixedsystem energy
    #non-dac
    layer5_e_mix = layer5_e_rram + + layer5_e_adc + layer5_e_mixgpu

    'layer6'
    # gpu energy
    layer6_e_gpu = gpu_e * layer_op[5]
    # mixedsystem rram energy
    # VMM
    layer6_e_rram = e_avg_rram * (layer_rram[5] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer6_e_adc = erg_ADC_x1 * 256 * 64 * 128
    # mixedsystem gpu energy
    layer6_e_mixgpu = gpu_e * layer_digital[5]
    # mixedsystem energy
    #non-dac
    layer6_e_mix = layer6_e_rram + + layer6_e_adc + layer6_e_mixgpu

    'layer7'
    # gpu energy
    layer7_e_gpu = gpu_e * layer_op[6]
    # mixedsystem rram energy
    # VMM
    layer7_e_rram = e_avg_rram * (layer_rram[6] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer7_e_adc = erg_ADC_x1 * 256 * 64 * 128
    # mixedsystem gpu energy
    layer7_e_mixgpu = gpu_e * layer_digital[6]
    # mixedsystem energy
    #non-dac
    layer7_e_mix = layer7_e_rram + + layer7_e_adc + layer7_e_mixgpu

    'layer8'
    # gpu energy
    layer8_e_gpu = gpu_e * layer_op[7]
    # mixedsystem rram energy
    # VMM
    layer8_e_rram = e_avg_rram * (layer_rram[7] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer8_e_adc = erg_ADC_x1 * 1024 * 128 * 1
    # mixedsystem gpu energy
    layer8_e_mixgpu = gpu_e * layer_digital[7]
    # mixedsystem energy
    #non-dac
    layer8_e_mix = layer8_e_rram + + layer8_e_adc + layer8_e_mixgpu

    'layer9'
    # gpu energy
    layer9_e_gpu = gpu_e * layer_op[8]
    # mixedsystem rram energy
    # VMM
    layer9_e_rram = e_avg_rram * (layer_rram[8] / 2) * 2  # divided by 2 for MAC. times 2 for differential pair
    # ADC
    layer9_e_adc = erg_ADC_x1 * 1024 * 1
    # mixedsystem gpu energy
    layer9_e_mixgpu = gpu_e * layer_digital[8]
    # mixedsystem energy
    #non-dac
    layer9_e_mix = layer9_e_rram + + layer9_e_adc + layer9_e_mixgpu


    print(f'layer1_e_mix:{layer1_e_mix}')
    print(f'layer1_e_rram:{layer1_e_rram}')
    print(f'layer1_e_mixgpu:{layer1_e_mixgpu}')
    print(f' e_cim_adc_layer1:{ layer1_e_adc}')
    print(f'layer2_e_mix:{layer2_e_mix}')
    print(f'layer2_e_rram:{layer2_e_rram}')
    print(f'layer2_e_mixgpu:{layer2_e_mixgpu}')
    print(f' layer2_e_adc:{ layer2_e_adc}')
    print(f'layer3_e_mix:{layer3_e_mix}')
    print(f'layer3_e_rram:{layer3_e_rram}')
    print(f'layer3_e_mixgpu:{layer3_e_mixgpu}')
    print(f' layer3_e_adc:{ layer3_e_adc}')
    print(f'layer4_e_mix:{layer4_e_mix}')
    print(f'layer4_e_rram:{layer4_e_rram}')
    print(f'layer4_e_mixgpu:{layer4_e_mixgpu}')
    print(f' layer4_e_adc:{ layer4_e_adc}')
    print(f'layer5_e_mix:{layer5_e_mix}')
    print(f'layer5_e_rram:{layer5_e_rram}')
    print(f'layer5_e_mixgpu:{layer5_e_mixgpu}')
    print(f' layer5_e_adc:{ layer5_e_adc}')
    print(f'layer6_e_mix:{layer6_e_mix}')
    print(f'layer6_e_rram:{layer6_e_rram}')
    print(f'layer6_e_mixgpu:{layer6_e_mixgpu}')
    print(f' layer6_e_adc:{ layer6_e_adc}')
    print(f'layer7_e_mix:{layer7_e_mix}')
    print(f'layer7_e_rram:{layer7_e_rram}')
    print(f'layer7_e_mixgpu:{layer7_e_mixgpu}')
    print(f' layer7_e_adc:{ layer7_e_adc}')
    print(f'layer8_e_mix:{layer8_e_mix}')
    print(f'layer8_e_rram:{layer8_e_rram}')
    print(f'layer8_e_mixgpu:{layer8_e_mixgpu}')
    print(f' layer8_e_adc:{ layer8_e_adc}')
    print(f'layer9_e_mix:{layer9_e_mix}')
    print(f'layer9_e_rram:{layer9_e_rram}')
    print(f'layer9_e_mixgpu:{layer9_e_mixgpu}')
    print(f' layer9_e_adc:{ layer9_e_adc}')
    print(f'layer1_e_gpu:{layer1_e_gpu}')
    print(f'layer2_e_gpu:{layer2_e_gpu}')
    print(f'layer3_e_gpu:{layer3_e_gpu}')
    print(f'layer4_e_gpu:{layer4_e_gpu}')
    print(f'layer5_e_gpu:{layer5_e_gpu}')
    print(f'layer6_e_gpu:{layer6_e_gpu}')
    print(f'layer7_e_gpu:{layer7_e_gpu}')
    print(f'layer8_e_gpu:{layer8_e_gpu}')
    print(f'layer9_e_gpu:{layer9_e_gpu}')



    #Total energy
    e_gpu_system = layer1_e_gpu + layer2_e_gpu + layer3_e_gpu + layer4_e_gpu + layer5_e_gpu + layer6_e_gpu + layer7_e_gpu + layer8_e_gpu + layer9_e_gpu
    e_gpu_system = 908*e_gpu_system
    e_gpu_early_exit_system = 908 * layer1_e_gpu + (908) * layer2_e_gpu + (908) * layer3_e_gpu + (908-47) * layer4_e_gpu + \
                                           (908-47-52) * layer5_e_gpu + (908-47-52-164) * layer6_e_gpu + (908-47-52-164-56) * layer7_e_gpu \
                                            + (908-47-52-164-56) * layer8_e_gpu + (0) * layer9_e_gpu
    e_mixed_system = layer1_e_mix + layer2_e_mix + layer3_e_mix + layer4_e_mix + layer5_e_mix + layer6_e_mix + layer7_e_mix +layer8_e_mix + layer9_e_mix
    e_mixed_system = 908*e_mixed_system

    #cosin_energy
    cos_e_gpu_system = [10,10,10,10,10,10,10,10,10,10,10,10]
    cos_e_rram_system = [1280,1280,1280,1280,2560,2560,2560,10240]

    #early_exit
    num =908
    #ternary
    e_mix_early_exit_ternary_conv_system = 908 * layer1_e_mix + (908) * layer2_e_mix + (908)  * layer3_e_mix + (908-47) * layer4_e_mix + \
                                           (908-47-52) * layer5_e_mix + (908-47-52-164) * layer6_e_mix + (908-47-52-164-56) * layer7_e_mix \
                                            + (908-47-52-164-56) * layer8_e_mix + (908-47-52-164-56-589) * layer9_e_mix
    cos_ternary_rram_num = 908*cos_e_rram_system[0]+(908)*cos_e_rram_system[1]+(908)*cos_e_rram_system[2]+(908-47)*cos_e_rram_system[3]+\
                          (908-47-52)*cos_e_rram_system[4]+ (908-47-52-164)*cos_e_rram_system[5]+ (908-47-52-164-56)*cos_e_rram_system[6]\
                          + (908-47-52-164-56)*cos_e_rram_system[7]
    cos_ternary_gpu_num = 908+(908)+(908)+(908-47)+(908-47-52)+(908-47-52-164)+(908-47-52-164-56)+ (908-47-52-164-56)
    e_cos_ternary_gpu = cos_ternary_gpu_num * 10 * gpu_e + cos_ternary_gpu_num * 10 * gpu_e + cos_ternary_rram_num * gpu_e #shift(+1) + 除法 + 范数（norm）
    e_cos_ternary_rram = e_avg_rram * (cos_ternary_rram_num / 2) * 2 #cosine multiply
    e_cam =  e_cos_ternary_gpu +  e_cos_ternary_rram
    e_cim =  e_mix_early_exit_ternary_conv_system

    e_cam =  e_cos_ternary_gpu +  e_cos_ternary_rram
    print(f'e_cam_rram_1:{e_avg_rram *cos_e_rram_system[7]}')
    print(f'e_cam_gpu_1:{cos_e_rram_system[7] *(gpu_e) + 10 * gpu_e + 10 * gpu_e }')
    print(f'e_cam_adc_1:{erg_ADC_x1*10}')
    e_cam_rram = e_cos_ternary_rram
    e_cam_gpu = e_cos_ternary_gpu
    e_cam_adc = erg_ADC_x1 * cos_ternary_gpu_num * 10

    e_cim =  e_mix_early_exit_ternary_conv_system
    e_cim_rram =  908 * layer1_e_rram + (908) * layer2_e_rram + (908) * layer3_e_rram + (908-47) * layer4_e_rram + \
                                           (908-47-52) * layer5_e_rram + (908-47-52-164) * layer6_e_rram + (908-47-52-164-56) * layer7_e_rram \
                                            + (908-47-52-164-56) * layer8_e_mix
    e_cim_gpu = 908 * layer1_e_mixgpu + (908) * layer2_e_mixgpu + (908) * layer3_e_mixgpu + (908-47) * layer4_e_mixgpu + \
                                           (908-47-52) * layer5_e_mixgpu + (908-47-52-164) * layer6_e_mixgpu + (908-47-52-164-56) * layer7_e_mixgpu \
                                            + (908-47-52-164-56) * layer8_e_mixgpu
    e_cim_adc = 908 * layer1_e_adc + (908) * layer2_e_adc + (908) * layer3_e_adc + (908-47) * layer4_e_adc + \
                                           (908-47-52) * layer5_e_adc + (908-47-52-164) * layer6_e_adc + (908-47-52-164-56) * layer7_e_adc \
                                            + (908-47-52-164-56) * layer8_e_adc
    print(f'e_cam:{e_cam},e_cam_rram:{e_cam_rram},e_cam_gpu:{e_cam_gpu},e_cam_adc:{e_cam_adc}')
    print(f'e_cim:{e_cim},e_cim_rram:{e_cim_rram},e_cim_gpu:{e_cim_gpu},e_cim_adc:{e_cim_adc}')


    e_mix_early_exit_ternary_system = e_mix_early_exit_ternary_conv_system + e_cos_ternary_gpu + e_cos_ternary_rram + e_cam_adc

    #rram_system
    e_rram_early_exit_system = 908 * layer1_e_rram + (908) * layer2_e_rram + (908) * layer3_e_rram + (908-47) * layer4_e_rram + \
                                           (908-13-160-162-188) * layer5_e_rram + (908-13-160-162-188-253) * layer6_e_rram + (908-13-160-162-188-253-1) * layer7_e_rram \
                                            + (908-13-160-162-188-253-1-131) * layer8_e_mix +  e_cos_ternary_rram 
    e_cos_ternary_rram_total = e_avg_rram * (908*cos_e_rram_system[0]+908*cos_e_rram_system[1]+(908)*cos_e_rram_system[2]+(908)*cos_e_rram_system[3]+\
                          (908)*cos_e_rram_system[4]+ 908*cos_e_rram_system[5]+ 908*cos_e_rram_system[6]+ 908*cos_e_rram_system[7]) 
    e_rram_early_total_system = 908 * layer1_e_rram + 908 * layer2_e_rram + (908) * layer3_e_rram + (908) * layer4_e_rram + \
                                           (908) * layer5_e_rram + (908) * layer6_e_rram + (908) * layer7_e_rram \
                                            + (908) * layer8_e_mix + e_cos_ternary_rram_total


    # print(f'gpu_system:{e_gpu_system},mix_system:{e_mixed_system}, e_gpu_early_exit_system:{e_gpu_early_exit_system},mix_early_exit_ternary_system:{e_mix_early_exit_ternary_system},'
    #       f'mix_early_exit_ternary_01_system:{e_mix_early_exit_ternary_01_system},mix_early_exit_ternary_05_system:{e_mix_early_exit_ternary_05_system},e_gpu_early_exit_system:{e_gpu_early_exit_system}')

    print(f'gpu_system:{e_gpu_system},mix_system:{e_mixed_system}, mix_early_exit_ternary_system:{e_mix_early_exit_ternary_system},e_gpu_early_exit_system:{e_gpu_early_exit_system}\
    ,e_rram_early_exit_system:{e_rram_early_exit_system},e_rram_early_total_system:{e_rram_early_total_system}')


if __name__ == '__main__':
    analog_system_energy(gpu='3090')
