package com.sparrowrecsys.nearline.flink;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.source.FileProcessingMode;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.net.URL;

class Rating{
    public String userId;
    public String movieId;
    public String rating;
    public String timestamp;
    public String latestMovieId;

    public Rating(String line){
        String[] lines = line.split(",");
        this.userId = lines[0];
        this.movieId = lines[1];
        this.rating = lines[2];
        this.timestamp = lines[3];
        this.latestMovieId = lines[1];
    }
}

public class RealTimeFeature {

    public void test() throws Exception {
        // set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment
                .getExecutionEnvironment();

        URL ratingResourcesPath = this.getClass().getResource("/webroot/sampledata/ratings.csv");

        // monitor directory, checking for new files
        TextInputFormat format = new TextInputFormat(
                new org.apache.flink.core.fs.Path(ratingResourcesPath.getPath()));

        DataStream<String> inputStream = env.readFile(
                format,
                ratingResourcesPath.getPath(),
                FileProcessingMode.PROCESS_CONTINUOUSLY,
                100);

        DataStream<Rating> ratingStream = inputStream.map(Rating::new);

        ratingStream.keyBy(rating -> rating.userId)
                .timeWindow(Time.seconds(1))
                .reduce(
                        (ReduceFunction<Rating>) (rating, t1) -> {
                            if (rating.timestamp.compareTo(t1.timestamp) > 0){
                                return rating;
                            }else{
                                return t1;
                            }
                        }
                ).addSink(new SinkFunction<Rating>() {
            @Override
            public void invoke(Rating value, Context context) {
                System.out.println("userId:" + value.userId + "\tlatestMovieId:" + value.latestMovieId);
            }
        });
        env.execute();
    }

    public static void main(String[] args) throws Exception {
        new RealTimeFeature().test();
    }
}
