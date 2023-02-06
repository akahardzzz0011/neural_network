clear;

layers = [784,30,10];
 
 w1 = randn(layers(2), layers(1));
 w2 = randn(layers(3), layers(2));
 b1 = randn(layers(2), 1);
 b2 = randn(layers(3), 1);
 
  load testi01;
 % load adjustments_01;
 
 kierros_testing = 10000;
 iteration = 100; 
 epoch = 30;
 eta = 8.0; %learning rate
 
  [w1,w2,b1,b2] = verkon_training_laskenta(w1,w2,b1,b2,iteration,epoch,eta);
 
 % [w1,w2,b1,b2] = golden_images(w1,w2,b1,b2,iteration,epoch,eta);
 % [w1,w2,b1,b2] = wrong_images(w1,w2,b1,b2,iteration,epoch,eta);
 % [w1,w2,b1,b2] = verkon_testaus(w1,w2,b1,b2,kierros_testing);
 % [w1,w2,b1,b2] = kuvan_demo(w1,w2,b1,b2);