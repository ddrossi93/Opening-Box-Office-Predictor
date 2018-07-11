//global var to access final data set
var data;

//get json for final data set
$.getJSON('https://api.myjson.com/bins/11ucm6', function(datajson) {
	console.log('in getjson time');
	data = datajson;
    console.log(data); 
});

$(document).ready(function() {
    $('#example').DataTable( {
    	"scrollY": 600,
        "scrollX": true,
        "ajax": {"url":"https://api.myjson.com/bins/11ucm6","dataSrc":""},
        "columns": [
            { "data": "Title" },
            { "data": "Studio" },
            { "data": " Opening " },
            { "data": "Date" },
            { "data": "Director" },
            { "data": " Budget " },
            { "data": "imdbID" },
            { "data": "Month" },
            { "data": "Year" },
            { "data": "Ticket_Price" },
            { "data": "Actor" },
            { "data": "Genre" },
            { "data": "Writer" },
            { "data": "Ratings" }
        ]
    } );
} );


