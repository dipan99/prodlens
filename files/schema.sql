-- ============================================
-- Gaming Electronics Product Catalog Schema
-- Based on Real Product Data (RTINGS.com)
-- Products: Monitors, Mice, Keyboards
-- ============================================

-- Brands table
CREATE TABLE brands (
    brand_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL UNIQUE,
    country_origin VARCHAR(100),
    website_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    brand_id INTEGER NOT NULL REFERENCES brands(brand_id),
    category_name VARCHAR(50) NOT NULL,
    release_year INTEGER,
    price DECIMAL(10, 2),
    product_link TEXT,
    avg_customer_rating DECIMAL(3, 2),
    ranking_general DECIMAL(3, 1),
    ranking_gaming DECIMAL(3, 1),
    ranking_office DECIMAL(3, 1),
    ranking_editing DECIMAL(3, 1),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_customer_rating CHECK (avg_customer_rating >= 0 AND avg_customer_rating <= 5),
    CONSTRAINT chk_ranking_general CHECK (ranking_general >= 0 AND ranking_general <= 10),
    CONSTRAINT chk_ranking_gaming CHECK (ranking_gaming >= 0 AND ranking_gaming <= 10),
    CONSTRAINT chk_ranking_office CHECK (ranking_office >= 0 AND ranking_office <= 10),
    CONSTRAINT chk_ranking_editing CHECK (ranking_editing >= 0 AND ranking_editing <= 10)
);

-- Monitor-specific specifications
CREATE TABLE monitor_specs (
    spec_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    
    -- Physical specifications
    size_inch DECIMAL(4, 1),
    curve_radius VARCHAR(20),
    wall_mount VARCHAR(50),
    borders_size_cm DECIMAL(4, 2),
    
    -- Performance ratings (RTINGS sub-scores)
    brightness_rating DECIMAL(3, 1),
    response_time_rating DECIMAL(3, 1),
    hdr_picture_rating DECIMAL(3, 1),
    sdr_picture_rating DECIMAL(3, 1),
    color_accuracy_rating DECIMAL(3, 1),
    
    -- Display technology
    pixel_type VARCHAR(20),
    subpixel_layout VARCHAR(50),
    backlight VARCHAR(50),
    color_depth_bit INTEGER,
    
    -- Contrast and local dimming
    native_contrast DECIMAL(8, 1),
    contrast_with_local_dimming DECIMAL(8, 1),
    local_dimming BOOLEAN,
    
    -- Brightness measurements (cd/m²)
    sdr_real_scene_cdm2 DECIMAL(6, 2),
    sdr_peak_100_window_cdm2 DECIMAL(6, 2),
    sdr_sustained_100_window_cdm2 DECIMAL(6, 2),
    hdr_real_scene_cdm2 DECIMAL(6, 2),
    hdr_peak_100_window_cdm2 DECIMAL(6, 2),
    hdr_sustained_100_window_cdm2 DECIMAL(6, 2),
    minimum_brightness_cdm2 DECIMAL(6, 2),
    
    -- Viewing angles (degrees)
    color_washout_from_left_degrees INTEGER,
    color_washout_from_right_degrees INTEGER,
    color_shift_from_left_degrees INTEGER,
    color_shift_from_right_degrees INTEGER,
    brightness_loss_from_left_degrees INTEGER,
    brightness_loss_from_right_degrees INTEGER,
    black_level_raise_from_left_degrees INTEGER,
    black_level_raise_from_right_degrees INTEGER,
    
    -- Color accuracy
    black_uniformity_native_std_dev DECIMAL(6, 3),
    white_balance_dE DECIMAL(4, 2),
    
    -- Refresh rate and resolution
    native_refresh_rate_hz INTEGER,
    max_refresh_rate_hz INTEGER,
    native_resolution VARCHAR(20),
    aspect_ratio VARCHAR(20),
    flicker_free BOOLEAN,
    
    -- Connectivity
    max_refresh_rate_over_hdmi_hz INTEGER,
    displayport VARCHAR(50),
    hdmi VARCHAR(50),
    usbc_ports INTEGER,
    
    UNIQUE(product_id)
);

-- Mouse-specific specifications
CREATE TABLE mouse_specs (
    spec_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    
    -- Physical specifications
    coating VARCHAR(20),
    length_mm DECIMAL(5, 1),
    width_mm DECIMAL(5, 1),
    height_mm DECIMAL(5, 1),
    grip_width_mm DECIMAL(5, 1),
    default_weight_gm DECIMAL(6, 2),
    weight_distribution VARCHAR(20),
    
    -- Ergonomics
    ambidextrous VARCHAR(30),
    left_handed_friendly BOOLEAN,
    finger_rest BOOLEAN,
    
    -- Buttons and controls
    total_number_of_buttons INTEGER,
    number_of_side_buttons INTEGER,
    profile_switching_button BOOLEAN,
    scroll_wheel_type VARCHAR(50),
    
    -- Connectivity and power
    connectivity VARCHAR(20),
    battery_type VARCHAR(30),
    maximum_of_paired_devices VARCHAR(20),
    cable_length_m DECIMAL(4, 2),
    
    -- Technical specifications
    mouse_feet_material VARCHAR(50),
    switch_type VARCHAR(20),
    switch_model VARCHAR(100),
    
    -- Software compatibility
    software_windows_compatibility BOOLEAN,
    software_macos_compatibility BOOLEAN,
    
    UNIQUE(product_id)
);

-- Keyboard-specific specifications
CREATE TABLE keyboard_specs (
    spec_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    
    -- Physical specifications
    size VARCHAR(30),
    height_cm DECIMAL(4, 1),
    width_cm DECIMAL(5, 1),
    depth_cm DECIMAL(4, 1),
    depth_with_wrist_rest_cm DECIMAL(4, 1),
    weight_kg DECIMAL(6, 3),
    
    -- Build quality
    keycap_material VARCHAR(20),
    
    -- Ergonomics
    curved_or_angled BOOLEAN,
    split_keyboard BOOLEAN,
    
    -- Hardware customizability
    replaceable_cherry_stabilizers BOOLEAN,
    switch_stem_shape VARCHAR(50),
    mechanical_switch_compatibility VARCHAR(50),
    magnetic_switch_compatibility VARCHAR(50),
    
    -- Backlighting
    backlighting BOOLEAN,
    rgb BOOLEAN,
    per_key_backlighting BOOLEAN,
    effects BOOLEAN,
    
    -- Cable and connectivity
    connectivity VARCHAR(20),
    detachable VARCHAR(50),
    connector_length_m DECIMAL(4, 2),
    connector_keyboard_side VARCHAR(30),
    bluetooth BOOLEAN,
    
    -- Extra features
    media_keys VARCHAR(20),
    trackpad_or_trackball BOOLEAN,
    scroll_wheel BOOLEAN,
    numpad BOOLEAN,
    windows_key_lock BOOLEAN,
    
    -- Typing quality
    key_spacing_mm DECIMAL(4, 1),
    average_loudness_dba DECIMAL(5, 2),
    
    -- Keystrokes
    pre_travel_mm DECIMAL(6, 3),
    total_travel_mm DECIMAL(6, 3),
    detection_ratio_percent DECIMAL(6, 3),
    
    -- Switch specifications
    switch_type VARCHAR(20),
    switch_feel VARCHAR(20),
    
    -- Software
    software_configuration_profiles VARCHAR(20),
    
    -- OS Compatibility
    windows_compatibility VARCHAR(30),
    macos_compatibility VARCHAR(30),
    linux_compatibility VARCHAR(30),
    
    UNIQUE(product_id)
);

-- Customer reviews
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    user_id INTEGER,
    rating INTEGER NOT NULL,
    review_title VARCHAR(255),
    review_text TEXT,
    source VARCHAR(100),  -- Added: captures review source (e.g., 'Amazon', 'BestBuy', 'RTINGS')
    verified_purchase BOOLEAN DEFAULT FALSE,
    helpful_count INTEGER DEFAULT 0,
    review_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_review_rating CHECK (rating >= 1 AND rating <= 5)
);

-- Professional ratings from tech reviewers (e.g., RTINGS)
CREATE TABLE professional_ratings (
    rating_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    reviewer_website VARCHAR(255),
    rating_general DECIMAL(3, 1),
    rating_gaming DECIMAL(3, 1),
    rating_office DECIMAL(3, 1),
    rating_editing DECIMAL(3, 1),
    pros TEXT,
    cons TEXT,
    summary TEXT,
    review_url VARCHAR(500),
    review_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_rating_general CHECK (rating_general >= 0 AND rating_general <= 10),
    CONSTRAINT chk_rating_gaming CHECK (rating_gaming >= 0 AND rating_gaming <= 10),
    CONSTRAINT chk_rating_office CHECK (rating_office >= 0 AND rating_office <= 10),
    CONSTRAINT chk_rating_editing CHECK (rating_editing >= 0 AND rating_editing <= 10)
);