

 function appendMovie2Row(rowId, movieName, movieId, year, rating, rateNumber, genres, baseUrl) {

    var genresStr = "";
    $.each(genres, function(i, genre){
        genresStr += ('<div class="genre"><a href="'+baseUrl+'collection.html?type=genre&value='+genre+'"><b>'+genre+'</b></a></div>');
    });


    var divstr = '<div class="movie-row-item" style="margin-right:5px">\
                    <movie-card-smart>\
                     <movie-card-md1>\
                      <div class="movie-card-md1">\
                       <div class="card">\
                        <link-or-emit>\
                         <a uisref="base.movie" href="./movie.html?movieId='+movieId+'">\
                         <span>\
                           <div class="poster">\
                            <img src="./posters/' + movieId + '.jpg" />\
                           </div>\
                           </span>\
                           </a>\
                        </link-or-emit>\
                        <div class="overlay">\
                         <div class="above-fold">\
                          <link-or-emit>\
                           <a uisref="base.movie" href="./movie.html?movieId='+movieId+'">\
                           <span><p class="title">' + movieName + '</p></span></a>\
                          </link-or-emit>\
                          <div class="rating-indicator">\
                           <ml4-rating-or-prediction>\
                            <div class="rating-or-prediction predicted">\
                             <svg xmlns:xlink="http://www.w3.org/1999/xlink" class="star-icon" height="14px" version="1.1" viewbox="0 0 14 14" width="14px" xmlns="http://www.w3.org/2000/svg">\
                              <defs></defs>\
                              <polygon fill-rule="evenodd" points="13.7714286 5.4939887 9.22142857 4.89188383 7.27142857 0.790044361 5.32142857 4.89188383 0.771428571 5.4939887 4.11428571 8.56096041 3.25071429 13.0202996 7.27142857 10.8282616 11.2921429 13.0202996 10.4285714 8.56096041" stroke="none"></polygon>\
                             </svg>\
                             <div class="rating-value">\
                              '+rating+'\
                             </div>\
                            </div>\
                           </ml4-rating-or-prediction>\
                          </div>\
                          <p class="year">'+year+'</p>\
                         </div>\
                         <div class="below-fold">\
                          <div class="genre-list">\
                           '+genresStr+'\
                          </div>\
                          <div class="ratings-display">\
                           <div class="rating-average">\
                            <span class="rating-large">'+rating+'</span>\
                            <span class="rating-total">/5</span>\
                            <p class="rating-caption"> '+rateNumber+' ratings </p>\
                           </div>\
                          </div>\
                         </div>\
                        </div>\
                       </div>\
                      </div>\
                     </movie-card-md1>\
                    </movie-card-smart>\
                   </div>';
    $('#'+rowId).append(divstr);
};


function addRowFrame(pageId, rowName, rowId, baseUrl) {
 var divstr = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 <a class="plainlink" title="go to the full list" href="'+baseUrl+'collection.html?type=genre&value='+rowName+'">' + rowName + '</a> \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + rowId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
     $(pageId).prepend(divstr);
};

function addRowFrameWithoutLink(pageId, rowName, rowId, baseUrl) {
 var divstr = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 <a class="plainlink" title="go to the full list" href="'+baseUrl+'collection.html?type=genre&value='+rowName+'">' + rowName + '</a> \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + rowId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
     $(pageId).prepend(divstr);
};

function addGenreRow(pageId, rowName, rowId, size, baseUrl) {
    addRowFrame(pageId, rowName, rowId, baseUrl);
    $.getJSON(baseUrl + "getrecommendation?genre="+rowName+"&size="+size+"&sortby=rating", function(result){
        $.each(result, function(i, movie){
          appendMovie2Row(rowId, movie.title, movie.movieId, movie.releaseYear, movie.averageRating.toPrecision(2), movie.ratingNumber, movie.genres,baseUrl);
        });
    });
};

function addRelatedMovies(pageId, containerId, movieId, baseUrl){

    var rowDiv = '<div class="frontpage-section-top"> \
                <div class="explore-header frontpage-section-header">\
                 Related Movies \
                </div>\
                <div class="movie-row">\
                 <div class="movie-row-bounds">\
                  <div class="movie-row-scrollable" id="' + containerId +'" style="margin-left: 0px;">\
                  </div>\
                 </div>\
                 <div class="clearfix"></div>\
                </div>\
               </div>'
    $(pageId).prepend(rowDiv);

    $.getJSON(baseUrl + "getsimilarmovie?movieId="+movieId+"&size=16&model=genreSimilarity", function(result){
            $.each(result, function(i, movie){
              appendMovie2Row(containerId, movie.title, movie.movieId, movie.releaseYear, movie.averageRating.toPrecision(2), movie.ratingNumber, movie.genres,baseUrl);
            });
    });
}

function addMovieDetails(containerId, movieId, baseUrl) {

    $.getJSON(baseUrl + "getmovie?id="+movieId, function(movieObject){
        var genres = "";
        $.each(movieObject.genres, function(i, genre){
                genres += ('<span><a href="'+baseUrl+'collection.html?type=genre&value='+genre+'"><b>'+genre+'</b></a>');
                if(i < movieObject.genres.length-1){
                    genres+=", </span>";
                }else{
                    genres+="</span>";
                }
        });

        var movieDetails = '<div class="row movie-details-header movie-details-block">\
                                        <div class="col-md-2 header-backdrop">\
                                            <img alt="movie backdrop image" height="250" src="./posters/'+movieObject.movieId+'.jpg">\
                                        </div>\
                                        <div class="col-md-9"><h1 class="movie-title"> '+movieObject.title+' </h1>\
                                            <div class="row movie-highlights">\
                                                <div class="col-md-2">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Release Year</div>\
                                                        <div> '+movieObject.releaseYear+' </div>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-3">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> MovieLens predicts for you</div>\
                                                        <div> 5.0 stars</div>\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading"> Average of '+movieObject.ratingNumber+' ratings</div>\
                                                        <div> '+movieObject.averageRating.toPrecision(2)+' stars\
                                                        </div>\
                                                    </div>\
                                                </div>\
                                                <div class="col-md-6">\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Genres</div>\
                                                        '+genres+'\
                                                    </div>\
                                                    <div class="heading-and-data">\
                                                        <div class="movie-details-heading">Links</div>\
                                                        <a target="_blank" href="http://www.imdb.com/title/tt'+movieObject.imdbId+'">imdb</a>,\
                                                        <span><a target="_blank" href="http://www.themoviedb.org/movie/'+movieObject.tmdbId+'">tmdb</a></span>\
                                                    </div>\
                                                </div>\
                                            </div>\
                                        </div>\
                                    </div>'
        $("#"+containerId).prepend(movieDetails);
    });
};





