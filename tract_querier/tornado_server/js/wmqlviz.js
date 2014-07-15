/*
$(function() {
	
	// create the sliders for the 2D sliders
	$("#yellow_slider").slider({
		slide: volumeslicingX
	});
	$("#yellow_slider .ui-slider-handle").unbind('keydown');
	
	$("#red_slider").slider({
		slide: volumeslicingY
	});
	$("#red_slider .ui-slider-handle").unbind('keydown');
	
	$("#green_slider").slider({
		slide: volumeslicingZ
	});
	$("#green_slider .ui-slider-handle").unbind('keydown');
	
});
*/
function init_viewer2d() {

	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];

	// X Slice
	if (sliceX) {
		sliceX.destroy();
	}
	sliceX = new X.renderer2D();
	sliceX.container = 'sliceX';
	sliceX.orientation = 'X';
	sliceX.init();
	sliceX.add(volume);
	sliceX.render();
	sliceX.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceX), 100);
	};

	// Y Slice
	if (sliceY) {
		sliceY.destroy();
	}
	sliceY = new X.renderer2D();
	sliceY.container = 'sliceY';
	sliceY.orientation = 'Y';
	sliceY.init();
	sliceY.add(volume);
	sliceY.render();
	sliceY.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceY), 100);
	};

	// Z Slice
	if (sliceZ) {
		sliceZ.destroy();
	}
	sliceZ = new X.renderer2D();
	sliceZ.container = 'sliceZ';
	sliceZ.orientation = 'Z';
	sliceZ.init();
	sliceZ.add(volume);
	sliceZ.render();
	sliceZ.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceZ), 100);
	  };

	// update 2d slice sliders
        /*
	var dim = volume.dimensions;
	$("#yellow_slider").slider("option", "disabled", false);
	$("#yellow_slider").slider("option", "min", 0);
	$("#yellow_slider").slider("option", "max", dim[0] - 1);
	$("#yellow_slider").slider("option", "value", volume.indexX);

	$("#red_slider").slider("option", "disabled", false);
	$("#red_slider").slider("option", "min", 0);
	$("#red_slider").slider("option", "max", dim[2] - 1);
	$("#red_slider").slider("option", "value", volume.indexZ);

	$("#green_slider").slider("option", "disabled", false);
	$("#green_slider").slider("option", "min", 0);
	$("#green_slider").slider("option", "max", dim[1] - 1);
	$("#green_slider").slider("option", "value", volume.indexY);
        */
}//init_viewer2d()

// show labels on hover
function on2DHover(renderer) {

	// check that feature is enabled
	if (!_ATLAS_.hoverLabelSelect){
		return;
	}

	// get cursor position
	var mousepos = renderer.interactor.mousePosition;
	var ijk = renderer.xy2ijk(mousepos[0], mousepos[1]);
        $('#anatomy_caption').html("");
        console.info(ijk)
	if (!ijk) {
		return;
	}

	//
	var orientedIJK = ijk.slice();
	orientedIJK[0] = ijk[0];
	orientedIJK[1] = ijk[2];
	orientedIJK[2] = ijk[1];

	var volume = _ATLAS_.currentVolume;
	
	// get the number associated with the label
	var labelvalue = volume.labelmap.image[orientedIJK[0]][orientedIJK[1]][orientedIJK[2]];
	if (!labelvalue || labelvalue == 0) {
		return;
	}
	
	var labelname = volume.labelmap.colortable.get(labelvalue)[0];
	var _r = parseInt(volume.labelmap.colortable.get(labelvalue)[1] * 255, 10);
	var _g = parseInt(volume.labelmap.colortable.get(labelvalue)[2] * 255, 10);
	var _b = parseInt(volume.labelmap.colortable.get(labelvalue)[3] * 255, 10);
        $('#anatomy_caption').html("");
	$('#anatomy_caption').html(labelname);
	$('#anatomy_caption').css('color', 'rgb( ' + _r + ',' + _g + ',' + _b + ' )' );
}


sliceX = null;
sliceY = null;
sliceZ = null;

function init_volume(filename, colortable) {
  volume = new X.volume();
  volume.file = filename;
  volume.labelmap.file = filename;
  volume.labelmap.colortable.file = colortable;

  _ATLAS_ = {};
  _ATLAS_.volumes = {};
  _ATLAS_.currentMesh = 0;
  _ATLAS_.meshOpacity = 0.9;
  _ATLAS_.labelOpacity = 0.5;
  _ATLAS_.hover = null;
  _ATLAS_.hoverLabelSelect = true;

  _ATLAS_.currentVolume = volume;
  _ATLAS_.volumes[_ATLAS_.currentVolume]  = volume
};

function init_websocket(host, tract_download_host) {
  console.info("websocket start");

  _tracts_ = {};

  $(document).ready(function () {
    _WS_ = new WebSocket(host);
    console.info("Activating stuff");
    _WS_.onopen = function () {
        console.info("websocket engage");
    };

    _WS_.onmessage = function(evt){
        var ev = JSON.parse(evt.data);
        if (ev['receiver'] == 'tract') {
          var name = ev['name'];
          var file = ev['file'];
          var action = ev['action'];
          if (action == 'add') {
            if (name in _tracts_) {
              console.info("Removing tract " + name);
              _tracts_gui_.remove(_tracts_[name].control);
              threeDRenderer.remove(_tracts_[name]);
            };

            delete _tracts_[name];

            _tracts_[name] = new X.fibers();
            _tracts_[name].file = 'files/' + file;
            _tracts_[name].caption = name;

            _tracts_[name].control = _tracts_gui_.add(_tracts_[name], 'visible');
            _tracts_[name].control.name(name);

            _tracts_[name].modified();
            
            threeDRenderer.add(_tracts_[name]);

          }

          if (action == 'remove') {
            if (name in _tracts_) {
              console.info("Removing tract " + name);
              _tracts_gui_.remove(_tracts_[name].control);
              threeDRenderer.remove(_tracts_[name]);
              delete _tracts_[name];
            }
          }

          if (action == 'download') {
                var iframe = document.createElement("iframe");
                iframe.src = tract_download_host + '/' + name;
                iframe.onload = function() {
                    console.log("Download trigger");
                };
                iframe.style.display = "none";
                document.body.appendChild(iframe);
          }
        }
    }
    _WS_.onclose = function () {
        console.info("connection closed");
    };
    window.location.hash = '#wmql_console';
  });
};


function init_terminal() {
  jQuery(function($, undefined) {
    $('#wmql_console').terminal("jsonrpc",
    {
        greetings: 'White Matter Query Language Console',
        name: 'wmql_console',
        prompt: '[WMQL] ',
        height: '19%',
        completion: function (terminal, string, callback) {
          var cmd = 'system.completion'
          var result = []

          $.jrpc("jsonrpc", cmd, [string], 
            function (json) {
              console.log(json)
              callback(json['result']);
            }, function(){}
          );
        },
        onBlur: function() {
          return false;
        }
    });
  });
};



window.onload = function() {

  threeDRenderer = new X.renderer3D();
  threeDRenderer.container = $( "#3d" )[0]
  threeDRenderer.init();

  // restore the original key bindings and reinitialize the interactor
  // so it doesn't catch anymore the 'r' key to reset the view
  //
  window.onkeydown = function (e) { return true; }
  threeDRenderer.interactor.config.KEYBOARD_ENABLED = false;
  threeDRenderer.interactor.init();

  threeDRenderer.onShowtime = function() {
    init_viewer2d();
  };
    
  // The GUI panel
  
  // indicate if the mesh was loaded
  var gui = new dat.GUI();

  var labelmapgui = gui.addFolder('Label Map');
  var labelMapVisibleController = labelmapgui.add(_ATLAS_.currentVolume, 'visible');

  _tracts_gui_ = gui.addFolder('Tracts');

  threeDRenderer.add(_ATLAS_.currentVolume)
  threeDRenderer.render();
}

