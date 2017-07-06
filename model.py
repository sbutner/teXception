'''
TODO:
* Finish model
* Utilities
* Training Script
* Inference Script

================================================================================
|                             Architecture:                                    |
|                Topic Model Output                                            |
|          Left <                                                              |
|    Stem <       Concatenate - Head - Sentence Reconstruction                 |
|          Right<                                                              |
|                 Parsing Output                                               |
|                                                                              |
|                                                                              |
|                              Sides:                                          |
|    [Char x Word x Sentences] -> CharConv [1 x Word x Sentence] ->            |
|    -> WordConv [1 x 1 x Sentence] -> SentConv [1 x 1 x 1] ->                 |
|    -> SentExp [1 x 1 x Sentence] -> WordExp [1 x Word x Sentence] ->         |
|    -> CharExp [Char x Word x Sentences]                                      |
|                                                                              |
================================================================================
'''





def text_tower():

    x = Dense(1024, input_dim, name='stem_dense')
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation('relu', name='stem_act')(x)

    left = Conv3D()(x)
    .
    .
    .
    left = BatchNormalization()(left)
    topic_output = Dense(1024, activation='relu',name='topic_output')(left)


    right = Conv3D()(x)
    .
    .
    .
    right = BatchNormalization()(right)

    parse_output = Dense(1024, activation='relu',name='parse_output')(right)

    x = layers.concatenate([left, right], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu', name='output_act')(x)

    main_output = Dense(-1, activation='softmax')(x)

    #compile model etc.,
    inputs = get_inputs()
    
    model = Model(inputs=[text_input],
                  outputs=[main_output, topic_output, parse_output],
                  loss_weights=[1.,0.8,0.8])
    

        

    
