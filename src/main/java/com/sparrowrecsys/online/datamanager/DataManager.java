package com.sparrowrecsys.online.datamanager;

import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.Utility;

import java.io.File;
import java.util.*;

/**
 * DataManager is an utility class, takes charge of all data loading logic.
 */

public class DataManager {
    //singleton instance
    private static volatile DataManager instance;
    HashMap<Integer, Movie> movieMap;
    HashMap<Integer, User> userMap;
    //genre reverse index for quick querying all movies in a genre
    HashMap<String, List<Movie>> genreReverseIndexMap;
    //LSH bucket reverse index, key is "hashTableIndex_bucketValue", value is the list of movieIds in that bucket
    HashMap<String, List<Integer>> lshBucketReverseIndexMap;

    //multi-probe LSH tuning: keep widening the probe radius until at least this many candidates are gathered
    private static final int LSH_MIN_CANDIDATE_SIZE = 10;
    //the largest neighbouring-bucket distance to probe on each hash-table axis before giving up
    private static final int LSH_MAX_PROBE_RADIUS = 2;

    private DataManager(){
        this.movieMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.genreReverseIndexMap = new HashMap<>();
        this.lshBucketReverseIndexMap = new HashMap<>();
        instance = this;
    }

    public static DataManager getInstance(){
        if (null == instance){
            synchronized (DataManager.class){
                if (null == instance){
                    instance = new DataManager();
                }
            }
        }
        return instance;
    }

    //load data from file system including movie, rating, link data and model data like embedding vectors.
    public void loadData(String movieDataPath, String linkDataPath, String ratingDataPath, String movieEmbPath,
                         String userEmbPath, String movieLshBucketPath, String movieRedisKey, String userRedisKey, String movieLshBucketRedisKey) throws Exception{
        loadMovieData(movieDataPath);
        loadLinkData(linkDataPath);
        loadRatingData(ratingDataPath);
        loadMovieEmb(movieEmbPath, movieRedisKey);
        loadMovieLshBucket(movieLshBucketPath, movieLshBucketRedisKey);
        if (Config.IS_LOAD_ITEM_FEATURE_FROM_REDIS){
            loadMovieFeatures("mf:");
        }

        loadUserEmb(userEmbPath, userRedisKey);
    }

    //load movie data from movies.csv
    private void loadMovieData(String movieDataPath) throws Exception{
        System.out.println("Loading movie data from " + movieDataPath + " ...");
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(movieDataPath))) {
            while (scanner.hasNextLine()) {
                String movieRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] movieData = movieRawData.split(",");
                if (movieData.length == 3){
                    Movie movie = new Movie();
                    movie.setMovieId(Integer.parseInt(movieData[0]));
                    int releaseYear = parseReleaseYear(movieData[1].trim());
                    if (releaseYear == -1){
                        movie.setTitle(movieData[1].trim());
                    }else{
                        movie.setReleaseYear(releaseYear);
                        movie.setTitle(movieData[1].trim().substring(0, movieData[1].trim().length()-6).trim());
                    }
                    String genres = movieData[2];
                    if (!genres.trim().isEmpty()){
                        String[] genreArray = genres.split("\\|");
                        for (String genre : genreArray){
                            movie.addGenre(genre);
                            addMovie2GenreIndex(genre, movie);
                        }
                    }
                    this.movieMap.put(movie.getMovieId(), movie);
                }
            }
        }
        System.out.println("Loading movie data completed. " + this.movieMap.size() + " movies in total.");
    }

    //load movie embedding
    private void loadMovieEmb(String movieEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading movie embedding from " + movieEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(movieEmbPath))) {
                while (scanner.hasNextLine()) {
                    String movieRawEmbData = scanner.nextLine();
                    String[] movieEmbData = movieRawEmbData.split(":");
                    if (movieEmbData.length == 2) {
                        Movie m = getMovieById(Integer.parseInt(movieEmbData[0]));
                        if (null == m) {
                            continue;
                        }
                        m.setEmb(Utility.parseEmbStr(movieEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }else{
            System.out.println("Loading movie embedding from Redis ...");
            Set<String> movieEmbKeys = RedisClient.getInstance().keys(embKey + "*");
            int validEmbCount = 0;
            for (String movieEmbKey : movieEmbKeys){
                String movieId = movieEmbKey.split(":")[1];
                Movie m = getMovieById(Integer.parseInt(movieId));
                if (null == m) {
                    continue;
                }
                m.setEmb(Utility.parseEmbStr(RedisClient.getInstance().get(movieEmbKey)));
                validEmbCount++;
            }
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        }
    }

    //load LSH bucket data generated by the offline Spark LSH model, then build the bucket reverse index.
    //each line of the file is "movieId:bucket0 bucket1 bucket2", one bucket value per hash table.
    private void loadMovieLshBucket(String movieLshBucketPath, String lshBucketKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            File lshBucketFile = new File(movieLshBucketPath);
            //the bucket file is produced by the offline Spark LSH job; skip gracefully if it has not been generated yet
            if (!lshBucketFile.exists()) {
                System.out.println("Movie LSH bucket file not found at " + movieLshBucketPath
                        + ", skip loading (embedding retrieval will fall back to full scan).");
                return;
            }
            System.out.println("Loading movie LSH bucket from " + movieLshBucketPath + " ...");
            int validBucketCount = 0;
            try (Scanner scanner = new Scanner(lshBucketFile)) {
                while (scanner.hasNextLine()) {
                    String rawBucketData = scanner.nextLine();
                    String[] bucketData = rawBucketData.split(":");
                    if (bucketData.length == 2) {
                        Movie m = getMovieById(Integer.parseInt(bucketData[0]));
                        if (null == m) {
                            continue;
                        }
                        setMovieLshBuckets(m, bucketData[1]);
                        validBucketCount++;
                    }
                }
            }
            System.out.println("Loading movie LSH bucket completed. " + validBucketCount + " movie buckets in total.");
        }else{
            System.out.println("Loading movie LSH bucket from Redis ...");
            Set<String> lshBucketKeys = RedisClient.getInstance().keys(lshBucketKey + "*");
            int validBucketCount = 0;
            for (String oneLshBucketKey : lshBucketKeys){
                String movieId = oneLshBucketKey.split(":")[1];
                Movie m = getMovieById(Integer.parseInt(movieId));
                if (null == m) {
                    continue;
                }
                setMovieLshBuckets(m, RedisClient.getInstance().get(oneLshBucketKey));
                validBucketCount++;
            }
            System.out.println("Loading movie LSH bucket completed. " + validBucketCount + " movie buckets in total.");
        }
    }

    //parse the space-separated bucket values of a movie, attach the bucket keys to the movie and add it to the reverse index.
    private void setMovieLshBuckets(Movie movie, String bucketStr){
        if (null == bucketStr || bucketStr.trim().isEmpty()){
            return;
        }
        String[] buckets = bucketStr.trim().split("\\s+");
        List<String> bucketKeys = new ArrayList<>();
        //prefix the bucket value with its hash table index so that buckets of different hash tables never collide
        for (int tableIndex = 0; tableIndex < buckets.length; tableIndex++){
            String bucketKey = tableIndex + "_" + buckets[tableIndex];
            bucketKeys.add(bucketKey);
            this.lshBucketReverseIndexMap.computeIfAbsent(bucketKey, k -> new ArrayList<>()).add(movie.getMovieId());
        }
        movie.setEmbBuckets(bucketKeys);
    }

    //get LSH nearest neighbor candidates of a movie, using the "OR" multi-bucket strategy combined with
    //multi-probe: a movie is a candidate if it shares any bucket with the input movie in any hash table,
    //and we additionally probe neighbouring buckets to recall near neighbours that fell across a boundary.
    public List<Movie> getLshCandidates(Movie movie){
        return getLshCandidates(movie, LSH_MIN_CANDIDATE_SIZE, LSH_MAX_PROBE_RADIUS);
    }

    /**
     * multi-probe LSH candidate retrieval. Standard LSH only looks at the bucket the movie hashes to, so a
     * near neighbour that fell just across a bucket boundary is missed. Multi-probe also inspects the
     * neighbouring buckets (bucket value +/- step on the same projection axis). The probe radius grows step
     * by step and stops early once enough candidates are gathered, so the extra cost is only paid when the
     * exact bucket is too sparse.
     * @param movie input movie
     * @param minCandidateSize stop widening the probe radius once the candidate pool reaches this size
     * @param maxProbeRadius the largest neighbouring-bucket distance to probe on each hash-table axis
     * @return movie candidates (pre-rank pool)
     */
    public List<Movie> getLshCandidates(Movie movie, int minCandidateSize, int maxProbeRadius){
        List<Movie> candidates = new ArrayList<>();
        if (null == movie || null == movie.getEmbBuckets()){
            return candidates;
        }
        HashSet<Integer> candidateIdSet = new HashSet<>();
        //expand the probe radius shell by shell; radius 0 is the movie's own bucket
        for (int radius = 0; radius <= maxProbeRadius; radius++){
            for (String bucketKey : movie.getEmbBuckets()){
                for (String probeKey : probeBucketKeys(bucketKey, radius)){
                    List<Integer> movieIds = this.lshBucketReverseIndexMap.get(probeKey);
                    if (null != movieIds){
                        candidateIdSet.addAll(movieIds);
                    }
                }
            }
            candidateIdSet.remove(movie.getMovieId());
            //already have enough candidates, no need to probe wider buckets
            if (candidateIdSet.size() >= minCandidateSize){
                break;
            }
        }
        for (Integer movieId : candidateIdSet){
            Movie candidate = getMovieById(movieId);
            if (null != candidate){
                candidates.add(candidate);
            }
        }
        return candidates;
    }

    //generate the probe bucket keys at exactly the given radius on the same hash-table axis.
    //radius 0 -> the bucket itself; radius r -> the two buckets r steps away on each side.
    //bucket values are integer-valued doubles (e.g. "-2.0"), so adding an integer step keeps the
    //"<tableIndex>_<value>" key format identical to the offline-generated one.
    private List<String> probeBucketKeys(String bucketKey, int radius){
        List<String> probeKeys = new ArrayList<>();
        if (radius == 0){
            probeKeys.add(bucketKey);
            return probeKeys;
        }
        int separatorIndex = bucketKey.indexOf('_');
        if (separatorIndex < 0){
            return probeKeys;
        }
        String tableIndex = bucketKey.substring(0, separatorIndex);
        String bucketValueStr = bucketKey.substring(separatorIndex + 1);
        double bucketValue;
        try {
            bucketValue = Double.parseDouble(bucketValueStr);
        } catch (NumberFormatException e){
            return probeKeys;
        }
        probeKeys.add(tableIndex + "_" + (bucketValue + radius));
        probeKeys.add(tableIndex + "_" + (bucketValue - radius));
        return probeKeys;
    }

    //load movie features
    private void loadMovieFeatures(String movieFeaturesPrefix) throws Exception{
        System.out.println("Loading movie features from Redis ...");
        Set<String> movieFeaturesKeys = RedisClient.getInstance().keys(movieFeaturesPrefix + "*");
        int validFeaturesCount = 0;
        for (String movieFeaturesKey : movieFeaturesKeys){
            String movieId = movieFeaturesKey.split(":")[1];
            Movie m = getMovieById(Integer.parseInt(movieId));
            if (null == m) {
                continue;
            }
            m.setMovieFeatures(RedisClient.getInstance().hgetAll(movieFeaturesKey));
            validFeaturesCount++;
        }
        System.out.println("Loading movie features completed. " + validFeaturesCount + " movie features in total.");
    }

    //load user embedding
    private void loadUserEmb(String userEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            System.out.println("Loading user embedding from " + userEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(userEmbPath))) {
                while (scanner.hasNextLine()) {
                    String userRawEmbData = scanner.nextLine();
                    String[] userEmbData = userRawEmbData.split(":");
                    if (userEmbData.length == 2) {
                        User u = getUserById(Integer.parseInt(userEmbData[0]));
                        if (null == u) {
                            continue;
                        }
                        u.setEmb(Utility.parseEmbStr(userEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            System.out.println("Loading user embedding completed. " + validEmbCount + " user embeddings in total.");
        }
    }

    //parse release year
    private int parseReleaseYear(String rawTitle){
        if (null == rawTitle || rawTitle.trim().length() < 6){
            return -1;
        }else{
            String yearString = rawTitle.trim().substring(rawTitle.length()-5, rawTitle.length()-1);
            try{
                return Integer.parseInt(yearString);
            }catch (NumberFormatException exception){
                return -1;
            }
        }
    }

    //load links data from links.csv
    private void loadLinkData(String linkDataPath) throws Exception{
        System.out.println("Loading link data from " + linkDataPath + " ...");
        int count = 0;
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(linkDataPath))) {
            while (scanner.hasNextLine()) {
                String linkRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = linkRawData.split(",");
                if (linkData.length == 3){
                    int movieId = Integer.parseInt(linkData[0]);
                    Movie movie = this.movieMap.get(movieId);
                    if (null != movie){
                        count++;
                        movie.setImdbId(linkData[1].trim());
                        movie.setTmdbId(linkData[2].trim());
                    }
                }
            }
        }
        System.out.println("Loading link data completed. " + count + " links in total.");
    }

    //load ratings data from ratings.csv
    private void loadRatingData(String ratingDataPath) throws Exception{
        System.out.println("Loading rating data from " + ratingDataPath + " ...");
        boolean skipFirstLine = true;
        int count = 0;
        try (Scanner scanner = new Scanner(new File(ratingDataPath))) {
            while (scanner.hasNextLine()) {
                String ratingRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = ratingRawData.split(",");
                if (linkData.length == 4){
                    count ++;
                    Rating rating = new Rating();
                    rating.setUserId(Integer.parseInt(linkData[0]));
                    rating.setMovieId(Integer.parseInt(linkData[1]));
                    rating.setScore(Float.parseFloat(linkData[2]));
                    rating.setTimestamp(Long.parseLong(linkData[3]));
                    Movie movie = this.movieMap.get(rating.getMovieId());
                    if (null != movie){
                        movie.addRating(rating);
                    }
                    if (!this.userMap.containsKey(rating.getUserId())){
                        User user = new User();
                        user.setUserId(rating.getUserId());
                        this.userMap.put(user.getUserId(), user);
                    }
                    this.userMap.get(rating.getUserId()).addRating(rating);
                }
            }
        }

        System.out.println("Loading rating data completed. " + count + " ratings in total.");
    }

    //add movie to genre reversed index
    private void addMovie2GenreIndex(String genre, Movie movie){
        if (!this.genreReverseIndexMap.containsKey(genre)){
            this.genreReverseIndexMap.put(genre, new ArrayList<>());
        }
        this.genreReverseIndexMap.get(genre).add(movie);
    }

    //get movies by genre, and order the movies by sortBy method
    public List<Movie> getMoviesByGenre(String genre, int size, String sortBy){
        if (null != genre){
            List<Movie> movies = new ArrayList<>(this.genreReverseIndexMap.get(genre));
            switch (sortBy){
                case "rating":movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));break;
                case "releaseYear": movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));break;
                default:
            }

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
        }
        return null;
    }

    //get top N movies order by sortBy method
    public List<Movie> getMovies(int size, String sortBy){
            List<Movie> movies = new ArrayList<>(movieMap.values());
            switch (sortBy){
                case "rating":movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));break;
                case "releaseYear": movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));break;
                default:
            }

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
    }

    //get movie object by movie id
    public Movie getMovieById(int movieId){
        return this.movieMap.get(movieId);
    }

    //get user object by user id
    public User getUserById(int userId){
        return this.userMap.get(userId);
    }
}
