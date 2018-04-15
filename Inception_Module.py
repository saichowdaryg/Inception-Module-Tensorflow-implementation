class nn:
    
    def __init__(self):
        self.weights={}
        self.biases={}
        
    def inception(self,inp,layer_name,out_depth=[64,128,32,32],intermed_depth=[64,16]):
        
        depth=inp.get_shape()[-1].value
        #creating weights 
        self.weights[layer_name+' 1x1']=tf.Variable(tf.truncated_normal([1,1,depth,out_depth[0]],stddev=0.1))
        self.weights[layer_name+' 1x1 for 3x3']=tf.Variable(tf.truncated_normal([1,1,depth,intermed_depth[0]]                                                                                ,stddev=0.1))
        self.weights[layer_name+' 1x1 for 5x5']=tf.Variable(tf.truncated_normal([1,1,depth,intermed_depth[1]]                                                                                ,stddev=0.1))
        self.weights[layer_name+' 3x3']=tf.Variable(tf.truncated_normal([3,3,intermed_depth[0],out_depth[1]]                                                                        ,stddev=0.1))
        self.weights[layer_name+' 5x5']=tf.Variable(tf.truncated_normal([5,5,intermed_depth[1],out_depth[2]]                                                                        ,stddev=0.1))
        self.weights[layer_name+' 1x1 after maxpool']=tf.Variable(tf.truncated_normal                                                                  ([1,1,depth,out_depth[3]],stddev=0.1))
        #creating biases
        self.biases[layer_name+' 1x1']=tf.Variable(tf.constant(0.1,shape=[out_depth[0]]))
        self.biases[layer_name+' 1x1 for 3x3']=tf.Variable(tf.constant(0.1,shape=[intermed_depth[0]]))
        self.biases[layer_name+' 1x1 for 5x5']=tf.Variable(tf.constant(0.1,shape=[intermed_depth[1]]))
        self.biases[layer_name+' 3x3']=tf.Variable(tf.constant(0.1,shape=[out_depth[1]]))
        self.biases[layer_name+' 5x5']=tf.Variable(tf.constant(0.1,shape=[out_depth[2]]))
        self.biases[layer_name+' 1x1 after maxpool']=tf.Variable(tf.constant(0.1,shape=[out_depth[3]]))
        #describing the network
        #strides for cnn=[stride over batch(i.e 1),stride over length (here 1),\
        #                stride over width (here 1),stride over channels (i.e 1)] i.e [1,1,1,1]
        #strides for max pooling:[1,1,1,1]
        c1=tf.nn.relu(tf.nn.conv2d(inp,self.weights[layer_name+' 1x1'],strides=[1,1,1,1],padding='SAME')+                      self.biases[layer_name+' 1x1'])
        c1_3=tf.nn.relu(tf.nn.conv2d(inp,self.weights[layer_name+' 1x1 for 3x3'],strides=[1,1,1,1],                                     padding='SAME')+self.biases[layer_name+' 1x1 for 3x3'])
        c1_5=tf.nn.relu(tf.nn.conv2d(inp,self.weights[layer_name+' 1x1 for 5x5'],strides=[1,1,1,1],                                     padding='SAME')+self.biases[layer_name+' 1x1 for 5x5'])
        c3=tf.nn.relu(tf.nn.conv2d(c1_3,self.weights[layer_name+' 3x3'],strides=[1,1,1,1],padding='SAME')                      +self.biases[layer_name+' 3x3'])
        c5=tf.nn.relu(tf.nn.conv2d(c1_5,self.weights[layer_name+' 5x5'],strides=[1,1,1,1],padding='SAME')+                      self.biases[layer_name+' 5x5'])
        mp=tf.nn.max_pool(inp,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
        cmp=tf.nn.relu(tf.nn.conv2d(mp,self.weights[layer_name+' 1x1 after maxpool'],strides=[1,1,1,1],                                   padding='SAME')+self.biases[layer_name+' 1x1 after maxpool'])
        return tf.concat([c1,c1_3,c1_5,cmp],axis=3)
    
    def dense(self,inp,layer_name,nodes):
        prev_channels=inp.get_shape()[-1].value
        self.weights[layer_name]=tf.Variable(tf.truncated_normal([prev_channels,nodes],stddev=0.1))
        self.bias[layer_name]=tf.Variable(tf.constant(0.0,shape=[nodes]))
        return tf.nn.relu(tf.matmul(inp,self.weights[layer_name]) + self.biases[layer_name])
    
    def flatten(self,inp):
        return tf.contrib.layers.flatten(inp)
    
    def logits(self,inp,nodes=10):
        prev_channels=inp.get_shape()[-1].value
        layer_name='logits'
        self.weights[layer_name]=tf.Variable(tf.truncated_normal([prev_channels,nodes],stddev=0.01))
        self.biases[layer_name]=tf.Variable(tf.constant(0.1,shape=[nodes]))
        return tf.matmul(inp,self.weights[layer_name]) + self.biases[layer_name]


