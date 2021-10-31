google_news_preprocessing_dict = {
    'africa': 'Africa',
    'alps': 'Alps',
    'amazon': 'Amazon',
    'america': 'America',
    'antarctica': 'Antarctica',
    'atlantis': 'Atlantis',
    'australia': 'Australia',
    'aztec': 'Aztec',
    'beijing': 'Beijing',
    'berlin': 'Berlin',
    'bermuda': 'Bermuda',
    'canada': 'Canada',
    'centaur': 'Centaur',
    'czech': 'Czech',
    'egypt': 'Egypt',
    'england': 'England',
    'europe': 'Europe',
    'france': 'France',
    'germany': 'Germany',
    'greece': 'Greece',
    'himalayas': 'Himalayas',
    'hollywood': 'Hollywood',
    'jupiter': 'Jupiter',
    'london': 'London',
    'mexico': 'Mexico',
    'moscow': 'Moscow',
    'new_york': 'New_York',
    'olympus': 'Olympus',
    'phoenix': 'Phoenix',
    'rome': 'Rome',
    'saturn': 'Saturn',
    'shakespeare': 'Shakespeare',
    'tokyo': 'Tokyo',
    'unicorn': 'Unicorn',
    'washington': 'Washington'
}


bugle = [0.18457031, 0.083984375, -0.09863281, -0.032226562, -0.45507812, -0.47460938, -0.080566406, 0.16113281, 0.04638672, 0.042236328, 0.072265625, -0.59375, 0.0078125, 0.15917969, 0.27929688, -0.17578125, 0.06982422, -0.021484375, -0.15136719, -0.51953125, -0.04736328, 0.29296875, 0.068847656, 0.16308594, -0.18847656, 0.20507812, -0.21386719, -0.10498047, -0.16113281, 0.20410156, -0.00024223328, 0.17675781, 0.03881836, -0.44335938, 0.20507812, -0.4453125, 0.23339844, 0.37109375, 0.025024414, -0.08984375, 0.22753906, -0.30273438, 0.24023438, -0.515625, 0.40625, -0.42382812, -0.29882812, 0.061035156, -0.51953125, -0.053710938, -0.33007812, -0.020751953, 0.028686523, 0.21484375, -0.07128906, -0.072265625, 0.036621094, -0.27929688, 0.106933594, -0.28125, -0.09863281, 0.20214844, 0.25976562, -0.075683594, -0.087402344, 0.1796875, -0.11669922, 0.24902344, -0.036865234, 0.25, 0.35742188, -0.013366699, 0.0029907227, -0.27539062, -0.19433594, -0.0045776367, -0.06689453, -0.14550781, 0.056884766, -0.10644531, 0.27539062, 0.012023926, -0.07373047, 0.009033203, -0.28125, 0.39648438, -0.06738281, -0.017089844, -0.1953125, 0.107421875, -0.21875, 0.33398438, -0.12109375, -0.16503906, -0.44921875, 0.25585938, 0.38867188, -0.25976562, 0.014282227, -0.7421875, 0.018798828, 0.15527344, 0.07128906, 0.25585938, -0.25585938, -0.14746094, 0.421875, 0.19726562, -0.050048828, -0.21777344, 0.18847656, 0.33203125, -0.25, -0.00680542, 0.27929688, 0.234375, 0.008972168, 0.25585938, -0.12988281, -0.05883789, 0.48046875, -0.14648438, -0.10644531, -0.15429688, 0.064941406, 0.006164551, 0.13476562, -0.040039062, 0.06689453, 0.47265625, -0.23632812, -0.10546875, -0.24804688, 0.13769531, 0.08154297, -0.111816406, -0.47851562, 0.03930664, -0.11328125, 0.21386719, -0.021606445, -0.11328125, -0.12109375, 0.20898438, -0.16992188, 0.14453125, 0.12792969, -0.4140625, -0.3203125, -0.03930664, 0.45507812, 0.39453125, -0.064941406, -0.11376953, -0.22949219, -0.11621094, -0.09765625, 0.025024414, 0.05053711, -0.29492188, 0.107421875, -0.083496094, 0.27539062, -0.31835938, -0.01977539, 0.103515625, 0.24121094, -0.45703125, 0.29882812, -0.00579834, 0.12695312, -0.27148438, -0.056884766, -0.107421875, -0.049316406, -0.018676758, -0.016601562, -0.31640625, -0.203125, -0.24316406, -0.041015625, 0.10449219, 0.29296875, 0.3359375, 0.17285156, 0.16699219, 0.13378906, -0.07080078, 0.24316406, 0.38476562, -0.05810547, -0.15625, -0.34765625, -0.14648438, 0.16894531, -0.34960938, 0.11767578, 0.265625, -0.13867188, 0.4296875, -0.03491211, -0.17480469, -0.38867188, -0.06933594, 0.16894531, 0.075683594, 0.052978516, 0.025390625, -0.2421875, -0.052734375, -0.055419922, -0.14648438, -0.38671875, -0.40820312, -0.002822876, -0.13671875, -0.099609375, -0.27929688, -0.328125, -0.23925781, 0.25, 0.546875, 0.32617188, -0.41992188, -0.24023438, 0.34765625, -0.17675781, 0.10986328, 0.46679688, 0.1796875, -0.46484375, -0.3984375, -0.25, 0.053466797, -0.09375, 0.024902344, -0.28710938, -0.421875, 0.02355957, 4.1484833e-05, 0.044189453, -0.45507812, -0.30664062, 0.33203125, -0.07763672, -0.22558594, 0.16699219, -0.23730469, -0.12988281, 0.19140625, -0.1640625, 0.31640625, 0.08886719, -0.36914062, 0.15039062, 0.07128906, 0.43359375, -0.43164062, -0.22167969, -0.53515625, -0.009460449, 0.20019531, 0.05444336, -0.09863281, -0.20605469, 0.21972656, -0.30273438, -0.20800781, -0.27539062, 0.24609375, -1.1980534e-05, -0.07910156, 0.21777344, -0.07324219, 0.26367188, -0.18945312, 0.09863281, 0.114746094, -0.12695312, 0.22851562, -0.025390625, -0.39453125, -0.009643555, 0.11767578, 0.091796875, -0.11816406, 0.19824219, 0.033203125, 0.12792969, 0.453125, 0.12792969, -0.26757812, -0.4765625, -0.27148438, 0.20996094, -0.03515625, 0.078125, -0.17675781, -0.203125, 0.3828125]

leprechaun = [0.38867188, 0.05883789, -0.12060547, 0.19335938, 0.10595703, -0.21191406, 0.107910156, 0.20800781, 0.21191406, -0.32617188, 0.010498047, -0.20996094, 0.26953125, -0.096191406, -0.1875, -0.095703125, -0.37890625, -0.14746094, 0.036865234, -0.26757812, 0.013793945, 0.020629883, 0.29882812, -0.19921875, -0.26171875, 0.33007812, -0.026245117, 0.22167969, 0.3359375, -0.22070312, -0.036132812, 0.05883789, 0.31054688, 0.203125, -0.10888672, -0.14550781, 0.47851562, 0.052001953, 0.0625, 0.3125, 0.115722656, -0.0032806396, 0.3984375, -0.203125, -0.12451172, -0.15234375, -0.58203125, 0.24316406, 0.609375, 0.09375, -0.53515625, 0.29882812, 0.13476562, -0.1875, 0.15234375, -0.10888672, 0.06542969, -0.69140625, 0.064453125, 0.060546875, 0.0018615723, 0.19042969, 0.061279297, 0.19140625, -0.16210938, -0.09375, -0.10839844, -0.08203125, -0.119628906, 0.12890625, 0.34179688, -0.024780273, -0.15234375, 0.061035156, 0.1796875, 0.25585938, 0.056152344, 0.01373291, 0.20703125, -0.036865234, 0.18359375, -0.15234375, -0.6328125, -0.28710938, -0.27929688, -0.25585938, -0.19628906, 0.265625, 0.5234375, -0.11767578, -0.11279297, 0.171875, -0.296875, -0.17675781, 0.19335938, 0.14648438, -0.00037193298, 0.07910156, 0.09667969, -0.16210938, -0.5234375, -0.22070312, 0.32617188, -0.0703125, 0.0859375, 0.115722656, 0.14257812, 0.22558594, 0.07470703, -0.24707031, 0.09765625, 0.06982422, -0.00021839142, 0.18945312, -0.071777344, 0.12451172, -0.14746094, -0.23242188, 0.3125, 0.024291992, -0.032714844, -0.20703125, -0.21777344, -0.45703125, -0.04638672, 0.18847656, 0.18066406, -0.041259766, -0.26171875, -0.16113281, -0.21484375, -0.078125, -0.041992188, 0.14257812, -0.15136719, 0.20410156, 0.052490234, -0.103027344, 0.15917969, 0.34570312, 0.21484375, -0.107910156, -0.051757812, -0.083984375, 0.15429688, 0.26171875, -0.107421875, -0.079589844, -0.012573242, 0.24707031, 0.05053711, 0.18554688, -0.296875, 0.30078125, 0.030273438, -0.083496094, -0.03564453, 0.007659912, -0.19238281, -0.018920898, 0.30273438, 0.14550781, 0.15820312, -0.19824219, 0.026489258, -0.19824219, 0.18945312, -0.024169922, -0.16308594, 0.15820312, 0.018676758, 0.049316406, 0.12207031, 0.18652344, -0.359375, -0.21484375, -0.096191406, 0.23535156, 0.19238281, -0.16992188, 0.03881836, -0.1328125, 0.11328125, -0.037353516, -0.095703125, -0.17871094, 0.083984375, -0.0024871826, 0.21972656, 0.004119873, 0.059326172, -0.075683594, -0.625, -0.08691406, 0.11816406, 0.12988281, -0.16699219, -0.15722656, -0.19335938, 0.0041503906, -0.13183594, -0.29296875, -0.052734375, -0.18164062, 0.2578125, 0.018432617, 0.2890625, 0.013793945, -0.04638672, -0.1328125, -0.20410156, -0.18261719, -0.016601562, -0.48632812, 0.052001953, 0.29296875, 0.2578125, -0.3359375, -0.0013046265, -0.12109375, 0.22070312, -0.33789062, 0.3828125, -0.31054688, -0.18261719, 0.3984375, 0.13476562, -0.18359375, -0.109375, -0.13964844, 0.103515625, 0.203125, -0.26171875, 0.17382812, 0.14941406, -0.07421875, -0.09423828, -0.32421875, 0.14746094, 0.16210938, 0.24609375, -0.015991211, 0.01550293, 0.09033203, 0.24707031, -0.04736328, -0.23632812, 0.25976562, 0.067871094, -0.51171875, -0.28515625, 0.025390625, 0.10986328, 0.37304688, -0.21972656, -0.022216797, -0.041748047, -0.092285156, 0.01361084, 0.010803223, 0.14746094, 0.24414062, -0.06225586, 0.10498047, 0.12792969, 0.16992188, -0.33398438, 0.07421875, -0.3046875, -0.14550781, -0.19335938, 0.2734375, -0.017333984, 0.18847656, 0.015136719, -0.15527344, 0.030883789, -0.025268555, -0.07910156, 0.24707031, 0.030029297, 0.25390625, 0.265625, 0.234375, -0.111816406, -0.17285156, -0.17578125, -0.12207031, -0.010864258, -0.083984375, 0.30078125, 0.19824219, -0.13964844, -0.24609375, 0.15136719, 0.34179688, 0.171875, -0.37890625, -0.10058594, -0.046875]

loch_ness = [-0.25390625, 0.1875, 0.100097656, 0.25, -0.06982422, 0.11230469, 0.10986328, 0.28125, 0.12792969, -0.12011719, -0.040039062, 0.29882812, -0.53125, -0.080078125, 0.34960938, -0.25, -0.2421875, 0.037109375, -0.017944336, -0.24023438, -0.29492188, -0.39453125, -0.08886719, 0.31054688, 0.19824219, -0.15429688, -0.296875, 0.23828125, 0.50390625, 0.3671875, -0.23828125, 0.43554688, -0.011962891, -0.1640625, -0.05078125, -0.30078125, -0.1484375, 0.111328125, 0.103027344, 0.038330078, -0.44140625, 0.03564453, -0.44140625, 0.043701172, -0.35351562, -0.2578125, 0.2421875, -0.064453125, 0.10107422, -0.10644531, -0.23242188, -0.018066406, 0.24609375, -0.080078125, -0.4375, -0.29101562, 0.09765625, -0.41796875, 0.087402344, -0.091308594, 0.38671875, 0.33007812, -0.09423828, -0.10546875, 0.005065918, -0.0625, -0.390625, -0.1875, -0.056884766, -0.0703125, -0.27734375, -0.43359375, 0.3828125, 0.25390625, -0.021240234, -0.033203125, 0.23828125, -0.13769531, 0.34960938, -0.46679688, 0.024536133, -0.42773438, -0.20019531, -0.03149414, -0.06689453, 0.20117188, 0.053222656, -0.33789062, 0.22265625, -0.27734375, 0.10839844, 0.022705078, 0.07763672, 0.42773438, 0.06933594, 0.20898438, 0.27539062, -0.16699219, 0.32617188, -0.10253906, -0.09423828, 0.19921875, 0.15917969, 0.15234375, 0.24707031, 0.23632812, -0.26367188, -0.078125, 0.17578125, -0.22949219, -0.21875, 0.026855469, 0.28710938, -0.0054016113, 0.27539062, 0.025390625, -0.4296875, -0.08642578, -0.1015625, 0.025512695, -0.3515625, -0.35742188, 0.22070312, -0.31054688, 0.21679688, -0.1875, -0.07763672, 0.07910156, -0.030029297, 0.15039062, -0.30078125, 0.28125, 0.029541016, 0.17578125, -0.092285156, -0.26757812, 0.25195312, -0.05810547, -0.091308594, 0.2734375, 0.12597656, 0.17675781, 0.24414062, -0.234375, 0.578125, 0.33398438, -0.18066406, 0.43164062, -0.19921875, -0.296875, -0.24609375, -0.265625, -0.17773438, -0.03881836, 0.16894531, 0.05810547, 0.453125, 0.051757812, -0.08691406, -0.14453125, -0.24414062, 0.33984375, -0.13867188, -0.44726562, 0.08935547, -0.18945312, 0.092285156, -0.08642578, -0.55078125, 0.47070312, -0.2109375, 0.26757812, 0.26757812, 0.43945312, -0.45507812, -0.5703125, 0.021484375, -0.3515625, -0.13085938, -0.09326172, -0.17480469, 0.05810547, -0.20117188, -0.41796875, -0.08544922, -0.27929688, 0.19726562, 0.00592041, 0.119628906, -0.06689453, -0.34179688, 0.08496094, 0.1171875, -0.20996094, 0.103027344, -0.09326172, -0.10253906, 0.11767578, 0.02319336, -0.42382812, 0.24902344, 0.05419922, -0.33398438, 0.04296875, 0.3203125, 0.14355469, 0.44726562, -0.27929688, -0.25585938, 0.30078125, -0.019042969, -0.18457031, 0.11669922, -0.03857422, -0.25390625, -0.1953125, 0.30664062, -0.34570312, 0.3046875, -0.4140625, 0.25976562, 0.12207031, 0.484375, -0.15527344, -0.06689453, 0.15917969, 0.1015625, 0.11230469, -0.17871094, -0.104003906, -0.27929688, -0.15625, 0.14941406, 0.03173828, 0.20214844, 0.29101562, 0.23144531, 0.234375, 0.44335938, -0.22851562, 0.026611328, -0.48046875, -0.32617188, -0.234375, 0.2578125, 0.020019531, 0.109375, 0.15039062, 0.23632812, -0.328125, 0.03466797, -0.30859375, 0.51953125, 0.421875, 0.24511719, 0.011230469, 0.44140625, 0.022949219, -0.171875, -0.2734375, 0.08886719, 0.24804688, 0.15136719, 0.171875, -0.068847656, 0.140625, -0.4921875, -0.084472656, -0.036621094, -0.14257812, -0.029785156, 0.091308594, 0.35742188, 0.22753906, 0.40625, 0.19726562, 0.13085938, -0.080566406, -0.021362305, 0.16894531, 0.27539062, -0.43164062, 0.21191406, 0.118652344, 0.55078125, -0.099121094, -0.5546875, 0.05908203, -0.041503906, 0.49609375, -0.26757812, 0.30664062, -0.5703125, 0.08886719, -0.030395508, -0.31445312, -0.06640625, -0.12597656, 0.2890625, -0.18652344]

platypus = [0.20898438, -0.041992188, -0.12988281, 0.14550781, -0.27539062, -0.390625, -0.019897461, 0.029174805, 0.265625, 0.48828125, 0.045654297, -0.053466797, -0.05834961, -0.09326172, -0.10644531, -0.08691406, -0.24609375, 0.03881836, 0.15527344, -0.038330078, 0.16015625, 0.10595703, -0.1171875, -0.16015625, -0.16699219, -0.114746094, -0.6875, 0.13574219, 0.40429688, -0.3515625, -0.12792969, -0.091308594, 0.11035156, 0.09863281, 0.10546875, 0.23242188, 0.041748047, 0.17871094, 0.035888672, 0.053955078, -0.4296875, 0.11425781, -0.18652344, 0.484375, -0.09326172, -0.39648438, 0.103027344, 0.119140625, -0.053710938, 0.15039062, -0.3203125, -0.46289062, 0.31835938, -0.11376953, 0.14453125, 0.16503906, -0.14257812, -0.17382812, 0.35742188, 0.21386719, 0.15234375, 0.10888672, 0.017089844, -0.18261719, -0.15332031, -0.26757812, 0.05859375, -0.15820312, -0.08154297, 0.079589844, 0.17578125, -0.13183594, 0.06738281, 0.14453125, -0.42382812, 0.041259766, -0.29296875, 0.02734375, -0.109375, -0.11767578, 0.10644531, -0.025268555, 0.19921875, 0.096191406, -0.092285156, 0.28710938, 0.07910156, -0.021118164, -0.014404297, -0.0058288574, -0.03491211, 0.2890625, -0.27539062, 0.24121094, -0.07373047, 0.08496094, -0.072753906, -0.0078125, -0.10986328, -0.14160156, 0.10449219, 0.20019531, 0.24121094, -0.032714844, 0.037597656, -0.22851562, 0.04663086, 0.3515625, -0.21679688, 0.076660156, -0.05908203, -0.12060547, 0.22363281, 0.22167969, 0.18164062, -0.31835938, -0.33203125, -0.1328125, -0.37304688, -0.034179688, 0.07421875, -0.087890625, -0.083984375, 0.09375, -0.38476562, 0.20703125, -0.17578125, 0.43554688, -0.032958984, 0.31054688, -0.29882812, -0.0003490448, -0.484375, 0.07128906, -0.15039062, 0.033935547, -0.07861328, 0.17773438, 0.21679688, 0.29882812, 0.21289062, -0.26757812, 0.35546875, 0.38671875, 0.16699219, 0.18847656, -0.28515625, 0.35546875, 0.014343262, -0.19433594, 0.064453125, -0.16503906, -0.46289062, 0.20117188, -0.28515625, -0.421875, -0.006134033, -0.061767578, -0.515625, -0.25, 0.00592041, 0.19628906, 0.328125, -0.026367188, -0.0033721924, -0.049560547, 0.31835938, 0.011657715, -0.18652344, 0.29101562, 0.064941406, 0.24511719, -0.014038086, 0.296875, 0.23144531, -0.072265625, 0.4375, -0.025878906, 0.1484375, -0.13085938, 0.07861328, 0.052490234, -0.16015625, -0.38476562, -0.23632812, 0.23242188, -0.1796875, 0.014709473, -0.29296875, 0.044677734, -0.03149414, -0.18554688, -0.14746094, -0.20019531, 0.011352539, 0.04296875, 0.087402344, -0.12988281, -0.3125, -0.22363281, 0.16601562, -0.053466797, -0.043701172, 0.20703125, -0.09716797, 0.28710938, 0.09277344, -0.12207031, 0.328125, -0.23730469, -0.28710938, -0.12890625, 0.039794922, -0.22851562, -0.032714844, -0.03857422, 0.34765625, 0.04272461, 0.375, -0.18359375, 0.25585938, 0.119628906, 0.049072266, -0.2265625, -0.16015625, -0.12597656, 0.24902344, 0.20410156, 0.10839844, 0.38476562, 0.08544922, -0.07421875, -0.037841797, 0.26757812, 0.2578125, -0.14550781, 0.15234375, -0.088378906, 0.5546875, -0.036376953, 0.25195312, -0.35351562, -0.20117188, 0.05126953, 0.2578125, 0.12011719, 0.20507812, 0.40820312, -0.06933594, -0.30664062, -0.453125, -0.13378906, 0.35546875, 0.18945312, 0.047851562, 0.1796875, 0.033447266, 0.0087890625, -0.3046875, -0.0390625, 0.38671875, 0.10595703, 0.021972656, 0.011169434, -0.033447266, 0.25976562, -0.09375, -0.26171875, 0.1484375, 0.08984375, -0.07324219, 0.09326172, 0.08544922, 0.16308594, 0.48828125, -0.28515625, 0.20703125, -0.08105469, 0.29882812, -0.18945312, -0.037841797, 0.050048828, 0.23730469, 0.27734375, 0.052490234, -0.39648438, 0.003112793, -0.21679688, 0.18261719, 0.078125, 0.08935547, 0.29492188, 0.044921875, 0.020507812, -0.25195312, -0.14648438, 0.18554688, 0.099121094, -0.099121094, -0.09667969]

robin = [-0.18066406, 0.3671875, -0.20214844, 0.1484375, 0.18457031, -0.359375, -0.19433594, 0.22753906, 0.2578125, 0.19628906, -0.092285156, -0.45117188, 0.048583984, -0.09667969, -0.0025024414, -0.047607422, -0.10205078, 0.05126953, -0.20019531, -0.30273438, -0.21386719, 0.035888672, 0.14160156, -0.32226562, -0.32226562, 0.091308594, -0.23339844, 0.44726562, 0.33007812, 0.16992188, 0.012756348, 0.12060547, -0.04345703, -0.34570312, 0.026245117, -0.08935547, -0.06933594, 0.16015625, 0.19140625, 0.4921875, 0.30078125, -0.18554688, 0.06689453, 0.24316406, 0.12792969, -0.28320312, 0.36523438, -0.33984375, 0.37109375, 0.16113281, -0.22558594, 0.028320312, 0.39453125, 0.25195312, 0.22070312, 0.043945312, -0.06298828, -0.25585938, 0.091308594, -0.099609375, 0.12060547, -0.18457031, -0.24707031, -0.27539062, -0.26367188, -0.22167969, -0.04736328, -0.24414062, -0.055419922, 0.057617188, 0.52734375, -0.123535156, -0.1640625, 0.06689453, -0.22851562, 0.064941406, -0.22558594, -0.41992188, 0.040527344, 0.14550781, -0.12695312, -0.12890625, 0.375, -0.24414062, 0.11328125, 0.23730469, 0.0036773682, -0.03955078, 0.390625, -0.31640625, -0.091308594, -0.03173828, -0.35546875, -0.22558594, -0.27539062, -0.053222656, 0.18457031, -0.296875, -0.14453125, -0.55078125, -0.052734375, -0.40039062, 0.2578125, 0.024536133, -0.27734375, -0.015136719, 0.12792969, 0.546875, -0.17480469, -0.20117188, -0.22753906, -0.13085938, 0.068847656, 0.34570312, -0.17382812, 0.041259766, 0.26367188, -0.203125, -0.18164062, 0.107421875, 0.09667969, -0.32421875, -0.14160156, -0.20507812, -0.28515625, 0.45898438, -0.068847656, 0.43164062, -0.043701172, 0.3359375, -0.33398438, -0.109375, 0.016113281, 0.025512695, 0.33789062, 0.060791016, -0.12792969, -0.052001953, 0.23242188, 0.18652344, 0.09716797, -0.03149414, 0.095703125, 0.14257812, 0.18945312, 0.11816406, -0.27734375, -0.008911133, -0.08691406, -0.37304688, 0.42382812, -0.04248047, -0.44140625, -0.0011444092, 0.0050354004, -0.2734375, 0.099609375, 0.19335938, -0.07763672, 0.13769531, -0.15625, -0.020019531, 0.15722656, -0.26171875, -0.19824219, -0.296875, 0.18554688, 0.068359375, 0.22363281, 0.19335938, -0.1640625, 0.18261719, -0.18652344, 0.25585938, -0.30273438, 0.03125, 0.1328125, 0.16503906, 0.13574219, -0.15332031, -0.30273438, -0.12988281, -0.13183594, 0.0029754639, -0.3046875, 0.18066406, -0.24902344, 0.18261719, -0.11767578, 0.41601562, -0.3203125, -0.29101562, -0.39453125, -0.03857422, 0.1796875, 0.17480469, 0.10253906, 0.043701172, -0.041259766, 0.23339844, -0.013793945, -0.22949219, -0.23144531, -0.051757812, -0.092285156, -0.07080078, 0.24023438, 0.08105469, 0.029785156, -0.05810547, -0.421875, 0.053710938, -0.38476562, -0.24316406, -0.34765625, 0.11279297, 0.38476562, -0.34179688, 0.04736328, -0.13183594, 0.20214844, -0.079589844, 0.22558594, -0.034179688, -0.13378906, -0.07519531, 0.041503906, 0.064453125, -0.012573242, 0.04663086, -0.24707031, -0.31835938, 0.010437012, -0.15039062, 0.20410156, 0.03149414, -0.064453125, -0.41210938, 0.41210938, 0.014770508, 0.19726562, -0.24511719, -0.30664062, -0.060546875, 0.18261719, 0.12792969, 0.10498047, 0.041503906, -0.07470703, 0.025878906, -0.140625, -0.17675781, 0.14453125, 0.020507812, -0.020874023, 0.106933594, -0.19238281, 0.11279297, -0.1328125, 0.13867188, 0.013122559, 0.088378906, -0.29882812, -0.3359375, 0.021606445, -0.040771484, -0.14257812, -0.020629883, -0.044189453, -0.23046875, -0.12988281, 0.20898438, -0.029907227, 0.029418945, 0.01940918, -0.22070312, 0.0050354004, 0.068847656, 0.0032806396, 0.010986328, 0.14648438, 0.234375, 0.13671875, 0.3203125, 0.265625, -0.22851562, 0.060546875, 0.22363281, 0.107421875, 0.14160156, -0.026611328, -0.08691406, 0.30859375, -0.21777344, 0.051513672, -0.12011719, -0.036376953, 0.079589844, 0.30664062, 0.23730469]

scorpion = [0.020019531, 0.103027344, -0.03540039, 0.34179688, -0.07421875, -0.09033203, -0.22851562, -0.12011719, -0.25, 0.30078125, 0.34765625, -0.5546875, -0.15527344, -0.19140625, -0.24707031, 0.06640625, -0.0115356445, -0.0006713867, -0.11621094, -0.022460938, -0.12158203, 0.14355469, 0.15136719, 0.17089844, -0.36523438, -0.37695312, 0.36328125, -0.08691406, 0.41796875, -0.095703125, -0.007446289, 0.31640625, -0.38867188, 0.16699219, -0.016601562, 0.027954102, 0.17578125, 0.27148438, 0.021240234, 0.16308594, 0.09033203, -0.03515625, 0.29882812, 0.27539062, -0.0014038086, -0.0390625, 0.30859375, 0.045410156, 0.13476562, -0.22851562, -0.20410156, 0.32226562, 0.3203125, -0.083496094, 0.123535156, -0.09863281, 0.15234375, 0.08544922, 0.13476562, 0.265625, 0.12207031, 0.08544922, 0.05810547, 0.41601562, -0.17773438, -0.25, -0.1796875, -0.43359375, -0.21386719, -0.14648438, -0.04736328, 0.020141602, -0.02734375, 0.028930664, -0.20019531, 0.115234375, -0.21386719, -0.28710938, -0.08105469, -0.15234375, 0.049072266, 0.13867188, -0.056884766, -0.40820312, -0.40234375, 0.171875, -0.43359375, -0.20117188, 0.017944336, -0.027832031, 0.140625, -0.00390625, -0.25585938, 0.26757812, -0.04711914, -0.036132812, 0.15917969, -0.3125, 0.06347656, -0.009216309, -0.0087890625, -0.43554688, 0.23632812, -0.3359375, -0.0625, 0.030395508, 0.14160156, 0.08203125, -0.104003906, -0.49414062, -0.12695312, -0.34960938, 0.28320312, 0.18945312, -0.14941406, 0.27539062, 0.21289062, -0.010498047, 0.043945312, 0.12792969, -0.12451172, -0.28125, -0.24316406, -0.21582031, -0.25195312, 0.18359375, 0.296875, -0.045898438, 0.34765625, 0.15429688, -0.45507812, 0.20898438, -0.44140625, 0.0005836487, -0.09033203, 0.20898438, 0.06982422, -0.107910156, 0.22753906, 0.58984375, 0.036376953, -0.12207031, -0.23144531, -0.04248047, 0.37304688, 0.056152344, -0.08642578, 0.11425781, 0.0078125, -0.18164062, 0.31835938, -0.07714844, -0.3828125, 0.022583008, 0.0703125, -0.20800781, -0.109375, -0.11376953, -0.03564453, 0.026733398, 0.34375, -0.035888672, 0.203125, -0.3828125, -0.29882812, -0.17773438, 0.39648438, -0.15039062, -0.24804688, 0.10058594, 0.12792969, 0.13574219, -0.03125, 0.22460938, -0.26953125, -0.16015625, 0.25976562, -0.14355469, -0.14257812, -0.088378906, -0.3359375, -0.10253906, -0.0119018555, 0.01940918, -0.390625, 0.16503906, 0.15234375, 0.10546875, -0.18652344, -0.07080078, -0.046875, 0.11328125, -0.28125, 0.060302734, 0.100097656, 0.24023438, 0.35351562, -0.068359375, -0.16894531, -0.07080078, 0.33789062, -0.21191406, 0.083496094, 0.20800781, 0.08886719, -0.22753906, 0.21679688, 0.23828125, -0.23828125, -0.080078125, -0.515625, -0.057128906, -0.19921875, -0.080078125, -0.18847656, 0.036132812, 0.39453125, 0.079589844, -0.20898438, -0.20117188, -0.025634766, -0.12695312, -0.048828125, 0.09667969, 0.030029297, 0.10058594, 0.13378906, -0.021240234, -0.067871094, 0.32226562, 0.0007247925, -0.28515625, 0.012573242, -0.022338867, 0.106933594, -0.16992188, -0.008544922, 0.31445312, 0.17089844, 0.049804688, 0.20703125, -0.39648438, -0.24511719, 0.021972656, 0.33007812, -0.099609375, -0.06347656, 0.033935547, 0.16015625, -0.0049438477, -0.080078125, -0.390625, 0.026733398, 0.006439209, 0.28515625, -0.19824219, -0.21191406, -0.3125, -0.083984375, -0.29296875, -0.029174805, 0.017700195, -0.03540039, -0.039794922, 0.13574219, 0.048583984, -0.12792969, -0.104003906, -0.09716797, -0.20117188, -0.13867188, 0.0072021484, 0.15625, 0.048339844, 0.421875, -0.11425781, -0.21875, -0.033447266, -0.063964844, -0.029174805, 0.10058594, 0.013000488, 0.21484375, 0.024902344, 0.13964844, -0.27148438, -0.017333984, -0.057128906, 0.053710938, 0.0025177002, -0.18847656, -0.06201172, -0.21289062, 0.11035156, 0.16796875, -0.06982422, 0.29296875, 0.18457031, -0.13183594, -0.119628906]

scuba_diver = [0.053710938, -0.03466797, -0.37695312, -0.012084961, -0.14550781, 0.4375, -0.18359375, -0.008850098, 0.28320312, -0.15332031, 0.3671875, -0.34765625, -0.28710938, -0.31445312, -0.10839844, 0.2890625, -0.071777344, 0.30664062, -0.009399414, 0.14550781, 0.16503906, 0.21777344, -0.08496094, -0.24414062, -0.18554688, -0.30273438, 0.03491211, 0.546875, 0.50390625, -0.171875, -0.024536133, 0.4921875, -0.27539062, -0.19042969, 0.026245117, -0.22363281, 0.00390625, 0.30664062, 0.12695312, -0.024902344, -0.17285156, -0.18652344, 0.11376953, 0.34765625, -0.013671875, -0.3125, -0.0234375, 0.002029419, 0.18261719, 0.25195312, -0.044433594, 0.17285156, 0.328125, -0.28710938, 0.092285156, -0.07910156, 0.34960938, -0.027954102, 0.23046875, 0.027832031, 0.008972168, 0.10839844, 0.0003604889, 0.12011719, 0.03515625, 0.40625, -0.28710938, -0.15136719, 0.099609375, -0.36328125, -0.1484375, -0.20214844, -0.3203125, -0.07373047, -0.053466797, 0.04736328, 0.14746094, -0.37109375, -0.056152344, 0.08691406, 0.26953125, 0.125, 0.046875, -0.25976562, 0.011230469, 0.14453125, -0.4921875, 0.06201172, -0.125, -0.29296875, -0.17675781, -0.11425781, -0.26757812, 0.4453125, 0.22265625, -0.09375, 0.21484375, 0.2265625, 0.11279297, 0.14648438, -0.048583984, -0.20605469, 0.30078125, -0.114746094, 0.11328125, 0.043945312, 0.41015625, 0.009521484, 0.07373047, -0.021362305, 0.008483887, 0.033691406, -0.029296875, 0.13476562, -0.12695312, 0.037841797, -0.58984375, 0.203125, 0.019165039, -0.1484375, 0.17773438, 0.03491211, 0.23242188, -0.114746094, 0.12597656, -0.14648438, 0.375, -0.10644531, -0.11376953, 0.45117188, -0.15039062, -0.06347656, -0.33203125, -0.036132812, -0.11669922, -0.21386719, 0.0703125, -0.14648438, -0.12890625, 0.37304688, -0.5234375, 0.0115356445, 0.03466797, 0.125, 0.011962891, 0.32421875, -0.48828125, 0.34179688, 0.010681152, -0.14941406, 0.38085938, -0.0026550293, -0.2578125, 0.107910156, -0.18164062, 0.16796875, 0.10986328, -0.046875, -0.13769531, 0.15136719, 0.05859375, 0.43554688, 0.19140625, -0.25585938, 0.21777344, -0.23730469, 0.2265625, -0.20410156, 0.20605469, -0.18847656, -0.12060547, 0.15917969, -0.46289062, 0.056396484, -0.0059509277, -0.052001953, -0.087402344, 0.0053100586, -0.05444336, -0.35742188, -0.0073547363, -0.15136719, -0.22265625, 0.11035156, -0.25390625, 0.21289062, -0.07324219, -0.22070312, 0.096191406, -0.08300781, -0.2734375, 0.36132812, -0.15917969, 0.6328125, 0.100097656, 0.19824219, 0.018432617, -0.068359375, -0.15722656, -0.07910156, -0.080078125, 0.09277344, -0.43945312, -0.21582031, -0.068847656, -0.12695312, -0.234375, -0.21582031, -0.14355469, -0.47265625, -0.30859375, 0.049072266, 0.20996094, 0.13769531, 0.07373047, -0.0625, 0.18847656, -0.25976562, 0.2265625, -0.3203125, 0.3359375, -0.34179688, 0.14453125, -0.038085938, 0.15039062, -0.013793945, 0.107910156, -0.20117188, 0.14355469, -0.123535156, 0.23144531, -0.35351562, -0.41210938, -0.47460938, -0.010925293, 0.25, 0.48046875, -0.11669922, -0.18554688, -0.016113281, 0.29492188, -0.33789062, -0.106933594, 0.28515625, -0.012451172, 0.023803711, -0.265625, -0.29296875, -0.087402344, -0.484375, -0.22851562, -0.18359375, 0.10107422, 0.296875, 0.22558594, 0.087890625, 0.43554688, -0.26171875, -0.18359375, -0.20898438, 0.17773438, -0.21386719, 0.05444336, -0.20410156, 0.28710938, 0.34765625, -0.15234375, 0.018798828, -0.020874023, -0.05908203, -0.23925781, -0.100097656, -0.03149414, 0.6484375, -0.091796875, 0.22851562, -0.4375, -0.12451172, -0.16894531, -0.072265625, 0.03540039, -0.072265625, 0.38671875, 0.23632812, 0.40820312, 3.7431717e-05, -0.375, -0.14355469, 0.21972656, 0.69921875, 0.29492188, -0.47265625, -0.17578125, -0.39648438, 0.2109375, -0.27539062, 0.010192871, -0.10986328, 0.09033203, 0.16894531]

undertaker = [0.20410156, 0.12988281, -0.11376953, 0.091308594, 0.22070312, 0.19335938, -0.12695312, -0.016723633, 0.14648438, -0.080566406, 0.4140625, -0.23242188, -0.004760742, -0.020019531, -0.21582031, 0.18945312, -0.044433594, 0.5, 0.111816406, -0.06347656, 0.20605469, -0.15625, 0.3359375, 0.110839844, 0.1796875, -0.119140625, -0.1875, -0.049072266, 0.08496094, -0.24121094, -0.033447266, 0.20800781, -0.00077438354, 0.23828125, 0.048828125, -0.20507812, 0.484375, 0.25390625, -0.005218506, 0.123046875, 0.27539062, -0.390625, -0.029174805, -0.22167969, 0.11621094, -0.088378906, -0.21972656, -0.026367188, 0.14355469, 0.14355469, -0.012573242, 0.025756836, -0.25390625, -0.010253906, 0.00982666, -0.2109375, -0.22265625, 0.051513672, 0.07519531, 0.24707031, 0.08935547, -0.004699707, -0.33007812, 0.040527344, -0.10058594, 0.1953125, -0.33398438, -0.2265625, -0.22167969, -0.19726562, 0.119140625, 0.140625, 0.15625, 0.28125, -0.18652344, 0.25976562, 0.013061523, 0.1328125, 0.12792969, -0.12158203, 0.31054688, -0.20898438, -0.27929688, -0.005859375, 0.072753906, 0.32421875, 0.08935547, -0.071777344, 0.30078125, 0.17089844, 0.18554688, 0.06933594, -0.20898438, -0.080566406, -0.07470703, 0.09423828, -0.20605469, 0.1875, -0.119628906, -0.3984375, -0.13574219, -0.13867188, 0.10595703, -0.20898438, 0.11425781, -0.012573242, -0.107910156, 0.41015625, -0.11279297, 0.079589844, -0.004211426, -0.10595703, -0.17382812, -0.06542969, -0.15625, 0.33203125, -0.10205078, 0.059814453, 0.31835938, 0.359375, 0.036865234, -0.13378906, 0.23535156, -0.55859375, 0.06201172, 0.115234375, -0.12011719, -0.34375, 0.29882812, -0.05126953, -0.095703125, -0.25976562, -0.16015625, 0.04321289, 0.0074157715, 0.076171875, -0.5, 0.10595703, -0.15429688, 0.125, 0.05029297, -0.09863281, -0.024414062, 0.27539062, 0.23730469, 0.11230469, -0.59375, 0.03100586, -0.34375, -0.056152344, 0.088378906, -0.33398438, -0.23046875, 0.15917969, -0.13574219, -0.14941406, -0.1484375, 0.049804688, 0.031982422, 0.044433594, -0.115722656, 0.15429688, 0.083496094, 0.010131836, 0.3046875, -0.21875, 0.22460938, -0.23046875, -0.0390625, -0.02758789, 0.1328125, 0.08544922, 0.04272461, -0.17382812, -0.31445312, -0.036376953, 0.3046875, -0.12988281, 0.15625, -0.28320312, -0.13964844, -0.044433594, -0.11230469, -0.115234375, -0.14160156, 0.17285156, 0.22460938, 0.039794922, -0.18066406, 0.23046875, -0.048583984, 0.47265625, -0.08496094, 0.46875, 0.28710938, -0.22167969, -0.13769531, 0.007385254, 0.067871094, 0.03955078, -0.22167969, -0.119628906, -0.36328125, -0.16796875, 0.084472656, 0.06591797, 0.19433594, 0.05883789, 0.22460938, -0.10205078, -0.10107422, -0.055664062, 0.011108398, -0.13867188, -0.34179688, 0.0625, -0.35351562, -0.055419922, -0.09082031, -0.022460938, 0.31640625, 0.31445312, 0.05834961, -0.10644531, 0.09814453, 0.111328125, 0.22753906, -0.0859375, -0.17480469, -0.106933594, 0.25976562, -0.29882812, -0.24707031, -0.084472656, -0.05126953, -0.03466797, 0.36914062, 0.24414062, -0.28320312, 0.040527344, 0.27734375, -0.29101562, 0.171875, 0.024902344, 0.14257812, -0.43945312, -0.18554688, 0.39453125, 0.13085938, -0.21972656, -0.34765625, 0.30664062, 0.049072266, 0.076660156, 0.011047363, 0.14355469, -0.036865234, -0.16308594, -0.100097656, -0.095214844, -0.15429688, -0.018554688, -0.034423828, -0.016845703, -0.080566406, 0.27539062, -0.18359375, 0.10986328, -0.03149414, 0.048095703, -0.12890625, -0.265625, 0.056640625, 0.017822266, -0.061035156, -0.07373047, 0.111328125, 0.38476562, 0.19921875, 0.025268555, 0.012268066, -0.40234375, 0.10644531, -0.015625, -0.050048828, 0.14550781, -0.18847656, 0.021362305, 0.09472656, 0.29101562, -0.0073242188, 0.045410156, -0.39257812, 0.24902344, -0.119140625, 0.39453125, -0.34570312, 0.043945312, 0.0037078857, -0.016967773]

google_news_missing_words = {
    "bugle": bugle,
    "leprechaun": leprechaun,
    "loch_ness": loch_ness,
    "platypus": platypus,
    "robin": robin,
    "scorpion": scorpion,
    "scuba_diver": scuba_diver,
    "undertaker": undertaker
}
