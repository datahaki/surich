// code by gemini
package ch.alpine.surich.net.demo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class BytePairEncoding {
  public static final String END_OF_WORD = "</w>";
  public static final String UNK_TOKEN = "<unk>";

  public record Pair(String left, String right) {
    @Override
    public String toString() {
      return left + " " + right;
    }
  }

  private final List<Pair> merges = new ArrayList<>();
  private final Map<Pair, Integer> mergeRanks = new HashMap<>();
  private final Map<String, Integer> vocabToId = new LinkedHashMap<>();
  private final List<String> idToVocab = new ArrayList<>();

  public BytePairEncoding() {
    addToken(UNK_TOKEN);
  }

  private int addToken(String token) {
    Integer existing = vocabToId.get(token);
    if (existing != null) {
      return existing;
    }
    int id = idToVocab.size();
    vocabToId.put(token, id);
    idToVocab.add(token);
    return id;
  }

  /** Trains BPE merges on one or more text documents up to targetVocabSize. */
  public void train(List<String> corpus, int targetVocabSize) {
    Objects.requireNonNull(corpus, "corpus must not be null");
    if (targetVocabSize < vocabToId.size()) {
      throw new IllegalArgumentException("targetVocabSize (" + targetVocabSize + ") must be at least " + vocabToId.size());
    }
    merges.clear();
    mergeRanks.clear();
    // Map from space-joined symbol sequence -> occurrence count of that word
    Map<String, Integer> wordCounts = new HashMap<>();
    for (String doc : corpus) {
      if (doc == null || doc.isBlank()) {
        continue;
      }
      for (String rawWord : doc.trim().split("\\s+")) {
        List<String> symbols = wordToInitialSymbols(rawWord);
        for (String sym : symbols) {
          addToken(sym);
        }
        String key = String.join(" ", symbols);
        wordCounts.merge(key, 1, Integer::sum);
      }
    }
    while (idToVocab.size() < targetVocabSize) {
      Map<Pair, Integer> pairCounts = countPairs(wordCounts);
      if (pairCounts.isEmpty()) {
        break;
      }
      Pair bestPair = selectBestPair(pairCounts);
      String mergedSymbol = bestPair.left() + bestPair.right();
      int rank = merges.size();
      merges.add(bestPair);
      mergeRanks.put(bestPair, rank);
      addToken(mergedSymbol);
      wordCounts = applyMergeToCorpus(wordCounts, bestPair, mergedSymbol);
    }
  }

  /** Encodes new text into subword tokens. */
  public List<String> encodeTokens(String text) {
    if (text == null || text.isBlank()) {
      return List.of();
    }
    List<String> result = new ArrayList<>();
    for (String rawWord : text.trim().split("\\s+")) {
      List<String> symbols = new ArrayList<>(wordToInitialSymbols(rawWord));
      while (symbols.size() > 1) {
        int bestRank = Integer.MAX_VALUE;
        int bestIndex = -1;
        for (int i = 0; i < symbols.size() - 1; i++) {
          Pair candidate = new Pair(symbols.get(i), symbols.get(i + 1));
          Integer rank = mergeRanks.get(candidate);
          if (rank != null && rank < bestRank) {
            bestRank = rank;
            bestIndex = i;
          }
        }
        if (bestIndex == -1) {
          break;
        }
        String merged = symbols.get(bestIndex) + symbols.get(bestIndex + 1);
        List<String> nextSymbols = new ArrayList<>(symbols.size() - 1);
        for (int i = 0; i < symbols.size(); i++) {
          if (i == bestIndex) {
            nextSymbols.add(merged);
            i++; // Skip the second symbol of the merged pair
          } else {
            nextSymbols.add(symbols.get(i));
          }
        }
        symbols = nextSymbols;
      }
      result.addAll(symbols);
    }
    return result;
  }

  /** Encodes new text into learned vocabulary token IDs (falling back to UNK_TOKEN). */
  public List<Integer> encodeIds(String text) {
    List<String> tokens = encodeTokens(text);
    List<Integer> ids = new ArrayList<>(tokens.size());
    int unkId = vocabToId.get(UNK_TOKEN);
    for (String token : tokens) {
      ids.add(vocabToId.getOrDefault(token, unkId));
    }
    return ids;
  }

  public List<Pair> getMerges() {
    return Collections.unmodifiableList(merges);
  }

  public Map<String, Integer> getVocab() {
    return Collections.unmodifiableMap(vocabToId);
  }

  private static List<String> wordToInitialSymbols(String word) {
    List<String> symbols = new ArrayList<>(word.codePointCount(0, word.length()) + 1);
    word.codePoints().forEach(cp -> symbols.add(Character.toString(cp)));
    symbols.add(END_OF_WORD);
    return symbols;
  }

  private static Map<Pair, Integer> countPairs(Map<String, Integer> wordCounts) {
    Map<Pair, Integer> pairCounts = new HashMap<>();
    for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
      String[] symbols = entry.getKey().split(" ");
      int freq = entry.getValue();
      for (int i = 0; i < symbols.length - 1; i++) {
        Pair pair = new Pair(symbols[i], symbols[i + 1]);
        pairCounts.merge(pair, freq, Integer::sum);
      }
    }
    return pairCounts;
  }

  private static Pair selectBestPair(Map<Pair, Integer> pairCounts) {
    Map.Entry<Pair, Integer> best = null;
    for (Map.Entry<Pair, Integer> entry : pairCounts.entrySet()) {
      if (best == null) {
        best = entry;
        continue;
      }
      int freqCmp = Integer.compare(entry.getValue(), best.getValue());
      if (freqCmp > 0 || (freqCmp == 0 && entry.getKey().left().compareTo(best.getKey().left()) > 0)
          || (freqCmp == 0 && entry.getKey().left().equals(best.getKey().left()) && entry.getKey().right().compareTo(best.getKey().right()) > 0)) {
        best = entry;
      }
    }
    if (best == null) {
      throw new IllegalStateException("pairCounts must not be empty");
    }
    return best.getKey();
  }

  private static Map<String, Integer> applyMergeToCorpus(Map<String, Integer> wordCounts, Pair targetPair, String mergedSymbol) {
    Map<String, Integer> updated = new HashMap<>(wordCounts.size());
    for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
      String[] symbols = entry.getKey().split(" ");
      List<String> next = new ArrayList<>(symbols.length);
      boolean changed = false;
      int i = 0;
      while (i < symbols.length) {
        if (i < symbols.length - 1 && symbols[i].equals(targetPair.left()) && symbols[i + 1].equals(targetPair.right())) {
          next.add(mergedSymbol);
          i += 2;
          changed = true;
        } else {
          next.add(symbols[i]);
          i += 1;
        }
      }
      String nextKey = changed ? String.join(" ", next) : entry.getKey();
      updated.merge(nextKey, entry.getValue(), Integer::sum);
    }
    return updated;
  }

  public static void main(String[] args) {
    List<String> trainingData = List.of("low lower newest widest", "low lower newest widest", "newer widest low", "lowest newer widest");
    BytePairEncoding bpe = new BytePairEncoding();
    bpe.train(trainingData, 30);
    System.out.println("Learned merges (" + bpe.getMerges().size() + "):");
    for (int i = 0; i < bpe.getMerges().size(); i++) {
      System.out.println("  " + i + ": " + bpe.getMerges().get(i));
    }
    String unseen = "lower newest lowest";
    System.out.println("\nInput:  " + unseen);
    System.out.println("Tokens: " + bpe.encodeTokens(unseen));
    System.out.println("IDs:    " + bpe.encodeIds(unseen));
  }
}
