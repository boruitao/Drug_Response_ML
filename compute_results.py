import re

f_read = open('results.txt')
line = f_read.readline()
counter = 1
batch_num = int(counter/5) + 1
sum_train = 0
sum_test = 0
while line:
    # Only look at lines containing scores
    if line.endswith('min\n'):

        if counter %5 == 1: # Start of new batch
            # Update the batch_num
            batch_num = int(counter/5) + 1

            # Reset sum_train and sum_test
            sum_train = 0
            sum_test = 0

        # Add the line's train and test values to sum_train/sum_test
        train_MSE = re.findall("train=-[0-9]\.[0-9]{3}", line)[0]
        train_MSE = float(train_MSE[7:])
        sum_train += train_MSE
        
        test_MSE = re.findall("test=-[0-9]\.[0-9]{3}", line)[0]
        test_MSE = float(test_MSE[6:])
        sum_test += test_MSE

        if counter %5 == 0: # On last line of batch
            # Compute average train/test MSE for current batch
            avg_train = round(sum_train / 5, 4)
            avg_test = round(sum_test / 5, 4)
            print(str(batch_num) + '. ' + line[0:80])
            print('[' + str(avg_train) + ', ' + str(avg_test) + ']\n')

        line = f_read.readline()
        counter += 1

    else:
        line = f_read.readline()

f_read.close()