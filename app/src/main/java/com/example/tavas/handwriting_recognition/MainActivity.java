package com.example.tavas.handwriting_recognition;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;

import com.rm.freedrawview.FreeDrawView;
import com.rm.freedrawview.PathDrawnListener;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    final String MODEL_FILE = "file:///android_asset/mnist_graph.pb";
    final String INPUT_NODE = "input";
    final String OUTPUT_NODE = "output";

    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    FreeDrawView drawView;
    ImageButton eraseButton;
    TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setupTensorflow();
        getViews();
        setupListeners();
    }

    private void getViews () {
        drawView = findViewById(R.id.drawView);
        eraseButton = findViewById(R.id.eraseButton);
        resultText = findViewById(R.id.resultText);
    }

    private void setupListeners () {
        eraseButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                drawView.clearDraw();
                resultText.setText("");
            }
        });

        drawView.setOnPathDrawnListener(new PathDrawnListener() {
            @Override
            public void onPathStart() {}

            @Override
            public void onNewPathDrawn() {
                drawView.getDrawScreenshot(new FreeDrawView.DrawCreatorListener() {
                    @Override
                    public void onDrawCreated(Bitmap drawing) {
                        inferDrawing(drawing);
                    }

                    @Override
                    public void onDrawCreationError() {}
                });
            }
        });
    }

    private void setupTensorflow () {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    private void inferDrawing (Bitmap drawing) {
        Bitmap scaledDown = scaleDown(drawing, 28, true);
        float[] pixels = getPixelData(scaledDown);
        inferenceInterface.feed(INPUT_NODE, pixels, 784);
        inferenceInterface.run(new String[] {OUTPUT_NODE});

        float[] result = new float[10];
        inferenceInterface.fetch(OUTPUT_NODE, result);
        int index = getIndexOfLargest(result);

        resultText.setText("" + index);
    }

    public int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }

    public static Bitmap scaleDown(Bitmap realImage, float maxImageSize,
                                   boolean filter) {
        float ratio = Math.min(
                (float) maxImageSize / realImage.getWidth(),
                (float) maxImageSize / realImage.getHeight());
        int width = Math.round((float) ratio * realImage.getWidth());
        int height = Math.round((float) ratio * realImage.getHeight());

        Bitmap newBitmap = Bitmap.createScaledBitmap(realImage, width,
                height, filter);
        return newBitmap;
    }

    public float[] getPixelData(Bitmap bitmap) {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // Get 28x28 pixel data from bitmap
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] retPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff;
            retPixels[i] = 0xff - b;
        }
        return retPixels;
    }
}
