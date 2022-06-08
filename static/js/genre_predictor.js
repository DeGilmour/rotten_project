function postWithAjax(url, data, method) {
	$.ajax({
		type: "POST",
		url: url,
		data: data,
		success: function (response) {
			try {
				response = JSON.parse(response)
			} catch (e) {
				false
			}

			if (response.success === undefined && response.success != false) {
				if (method) return method(response)
			} else {
				return onErrorResponse(response)
			}
		},
		error: function (response) {
			console.log(response)
		}
	});
}


function guessMovie() {
	let movie_title = document.getElementById("movieTittle").value;
	let movie_info = document.getElementById("movieSynopsis").value;
	let data = postWithAjax('get-prediction', {
		'movie_title': movie_title,
		"movie_info": movie_info
	}, guessMovieCall)
	console.log(data)
}

function recomendMovie() {
	let movie_title_rec = document.getElementById("movie_title_rec").value;
	let data = postWithAjax('get-recomendation', {
		'movie_title': movie_title_rec
	}, recomendMovieCall)
	console.log(data)
}

function guessMovieCall(data) {
	let prediction_p = document.getElementById("prediction");
	console.log(data)
	prediction_p.innerHTML = data['prediction']
}

function recomendMovieCall(data) {
	let rec_movie = document.getElementById("rec-movie");
	console.log(data)
	rec_movie.innerHTML = data['recomendation']
    let data_frame = document.getElementsByClassName('dataframe')
    data_frame[0].classList.add('table')
}


$(document).ready(function () {
	$("#guess_btn").click(function () {
		$('.modal-backdrop').css("filter", "blur(200px)");
	});

});


        $(document).ready(function() {

            $('.modal.fade').each(function() {
                $(this).on('shown.bs.modal', function(e) {
                    $('body').toggleClass('modal-active');
                });
                $(this).on('hidden.bs.modal', function(e) {
                    $('body').toggleClass('modal-active');
                });
            });

            $('#atoresBtn').on('click', function() {
                let ator = $('#atores').val();
                $('#atoresPool').prepend('<span class="atorCard py-1 px-2 mx-1 bg-secondary text-white rounded-pill small">' + ator + '</span>');

                $('#atores').val("")
            });

            $("body").on("click", "span", function() {
                $(this).remove();
            });

            var arrData = [{
                'id': 0,
                'text': 'Action & Adventure'
            }, {
                'id': 1,
                'text': 'Comedy'
            }, {
                'id': 2,
                'text': 'Classics'
            }, {
                'id': 3,
                'text': 'Art House & International'
            }, {
                'id': 4,
                'text': 'Drama'
            }, {
                'id': 5,
                'text': 'Documentary'
            }, {
                'id': 6,
                'text': 'Animation'
            }, {
                'id': 7,
                'text': 'Horror'
            }, {
                'id': 8,
                'text': 'Kids & Family'
            }, {
                'id': 9,
                'text': 'Mystery & Suspense'
            }, {
                'id': 10,
                'text': 'Romance'
            }, {
                'id': 11,
                'text': 'Cult Movies'
            }, {
                'id': 12,
                'text': 'Science Fiction & Fantasy'
            }, {
                'id': 13,
                'text': 'Musical & Performing Arts'
            }, {
                'id': 14,
                'text': 'Western'
            }, {
                'id': 15,
                'text': 'Special Interest'
            }, {
                'id': 16,
                'text': 'Television'
            }]

            for (let index = 0; index < arrData.length; index++) {
                $('#selectGenre').append('<option value=' + arrData[index].id + '">' + arrData[index].text + '</option>')
            }
        });