# MNIST_NTOU
add some new on MINST

part 1    Make the train model

1.開啟cnn_train,it will import cnn_model file in this process.

2.cnn_model use train image to generate predict labels(predict labels是憑空產生的labels，其格式與train labels一樣)

3.將 predict labels 丟入"cross_entropy", 與train image 的 label 做比較，
  然後產生此model的正確率。

4.開始train(在此我們訓練次數設為２００）

5.每１０次印出目前model的正確率

6.跑完預設次數後，此model的變數不再更改。（就是訓練完了）

7.將test data利用該model，顯示出其正確率。

8.檢視該model正確率是否滿意，滿意則進行存取，反之再繼續train該model。

9.存取該model的變數，此處只有weight與bias。

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

part 2    讀任意圖片，藉由已建構的model，輸出該圖最有可能的數字。

1.

2.




cnn_model
