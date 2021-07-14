package com.jhgan.nlp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class writeText {

    public static void main(String[] args) throws IOException {
        // 출력 파일
        String outputFileName = new String("/home/jhgan/workspace/nlp-example/outputs/output.txt");
        File file = new File(outputFileName);
        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        writer.append("hello world!");
        writer.close();

    }
}
