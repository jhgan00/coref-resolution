
package com.jhgan.nlp;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CorefExample {

    Set<String> pronounTags = new HashSet<String>() {{
        add("PRP");
        add("PRP$");
        add("WP");
        add("WP$");
    }};

    public static void main(String[] args) throws IOException {

        String inputFileName = "inputs/TheKoreanEconomy2010.txt";
        String outputFileName = "outputs/TheKoreanEconomy2010-coref";

        com.jhgan.nlp.CorefExample corefExample = new com.jhgan.nlp.CorefExample();

        // 입력 파일
        System.out.print("Reading Input File: " + inputFileName + " ... ");
        List<String> paragraphs;
        try (Stream<String> lines = Files.lines(Paths.get(inputFileName))) {
            paragraphs = lines.collect(Collectors.toList());
        }
        int numParagraphs = paragraphs.size();
        System.out.println("done");

        // 출력 파일
        String textFileName = outputFileName + "-text.txt";
        File textFile = new File(textFileName);
        BufferedWriter textWriter = new BufferedWriter(new FileWriter(textFile));

        String posFileName = outputFileName + "-pos.txt";
        File posFile = new File(posFileName);
        BufferedWriter posWriter = new BufferedWriter(new FileWriter(posFile));

        String chunkFileName = outputFileName + "-chunk.txt";
        File chunkFile = new File(chunkFileName);
        BufferedWriter chunkWriter = new BufferedWriter(new FileWriter(chunkFile));

        // pipeline 생성
        System.out.print("Building CoreNLP pipeline ... ");
        Properties props = new Properties();

        // 토크나이징 -> 문장분리 -> POS -> 원형복원 -> NER -> 파싱 -> Co-reference
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, coref");
        props.put("coref.algorithm", "neural");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        System.out.println("done");

        for (int i = 0; i < numParagraphs; i++) {

            String paragraph = paragraphs.get(i);

            // 단락에 대해서 복원 실행 -> 문장별로
            ArrayList<List<String>>  result = corefExample.resolveText(pipeline, paragraph);

            for (String resolvedText : result.get(0)) {
                textWriter.append(resolvedText + System.lineSeparator());
            }

            for (String resolvedPOS : result.get(1)) {
                posWriter.append(resolvedPOS + System.lineSeparator());
            }

            for (String resolvedChunk : result.get(2)) {
                chunkWriter.append(resolvedChunk + System.lineSeparator());
            }

            System.out.println("progress: " + (i + 1) + " / " + numParagraphs);
        }

        textWriter.close();
        posWriter.close();
        chunkWriter.close();

    }


    private ArrayList<List<String>>  resolveText(StanfordCoreNLP pipeline, String text) throws IOException {

        // 파이프라인에서 coreference 태깅
        Annotation doc = new Annotation(text);
        pipeline.annotate(doc);

        Map<Integer, CorefChain> corefs = doc.get(CorefCoreAnnotations.CorefChainAnnotation.class);
        List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
        List<String> resolvedSentences = new ArrayList<String>();
        List<String> resolvedPOS = new ArrayList<String>();
        List<String> resolvedChunks = new ArrayList<String>();

        // 문장 순회
        for (CoreMap sentence : sentences) {

            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            List<String> resolved = new ArrayList<String>();
            List<String> posList = new ArrayList();

            // 토큰 순회
            for (CoreLabel token : tokens) {

                Integer corefClustId = token.get(CorefCoreAnnotations.CorefClusterIdAnnotation.class);
                CorefChain chain = corefs.get(corefClustId);
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);

                // 조건에 맞는 대명사가 아니면 패스
                if (chain == null || chain.getMentionsInTextualOrder().size() == 1 || !pronounTags.contains(pos)) {
                    resolved.add(token.word());
                    posList.add(pos);

                }

                // 길이 2 이상의 co-reference 체인이 존재하는 대명사라면
                else {

                    // 대표 멘션을 찾기
                    int sentINdx = chain.getRepresentativeMention().sentNum - 1;
                    CoreMap corefSentence = sentences.get(sentINdx);
                    List<CoreLabel> corefSentenceTokens = corefSentence.get(CoreAnnotations.TokensAnnotation.class);
                    CorefChain.CorefMention reprMent = chain.getRepresentativeMention();

                    // 대표 멘션의 길이가 1 이고 대명사인 경우가 있어서 해당 경우 처리
                    int reprMentLength = reprMent.endIndex - reprMent.startIndex;
                    if (reprMentLength == 1) {
                        CoreLabel label = corefSentenceTokens.get(reprMent.startIndex - 1);
                        String labelPos = label.get(CoreAnnotations.PartOfSpeechAnnotation.class);

                        // 대표 멘션이 대명사이면 패스
                        if (pronounTags.contains(labelPos)) {
                            resolved.add(token.word());
                            posList.add(token.get(CoreAnnotations.PartOfSpeechAnnotation.class));
                        }

                        // 대표 멘션이 대명사가 아니면 복원
                        else {
                            resolved.add(label.word());
                            posList.add(label.get(CoreAnnotations.PartOfSpeechAnnotation.class));
                        }
                    } else {
                        for (int i = reprMent.startIndex; i < reprMent.endIndex; i++) {
                            CoreLabel matchedLabel = corefSentenceTokens.get(i - 1);
                            String matchedWord = matchedLabel.word();
                            resolved.add(matchedWord);
                            posList.add(matchedLabel.get(CoreAnnotations.PartOfSpeechAnnotation.class));
                        }
                    }
                }
            }

            String[] wordArray = resolved.toArray(new String[0]);
            String[] posArray = posList.toArray(new String[0]);
            InputStream inputStream = new FileInputStream("en-chunker.bin");
            ChunkerModel chunkerModel = new ChunkerModel(inputStream);
            ChunkerME chunkerME = new ChunkerME(chunkerModel);

            //result array is the list of chunked word such as NP, VP, etc.
            String result[] = chunkerME.chunk(wordArray, posArray);

            String resolvedStr = "";
            for (String str : resolved) {
                resolvedStr += str + " ";
            }

            String resolvedPos = "";
            for (String str : posList) {
                resolvedPos += str + " ";
            }

            String resolvedChunk = "";
            for (String str : result) {
                resolvedChunk += str + " ";
            }

            resolvedSentences.add(resolvedStr);
            resolvedPOS.add(resolvedPos);
            resolvedChunks.add(resolvedChunk);

        }

        ArrayList<List<String>> listOLists = new ArrayList<List<String>>();
        listOLists.add(resolvedSentences);
        listOLists.add(resolvedPOS);
        listOLists.add(resolvedChunks);

        return listOLists;
    }
}