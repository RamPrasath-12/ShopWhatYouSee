import axios from 'axios';

const API_KEY = '58142d6f';
const BASE_URL = 'https://www.omdbapi.com/';

const VIDEO_SOURCES = [
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4"
];

const getVideo = (index) => VIDEO_SOURCES[index % VIDEO_SOURCES.length];

// Helper to fetch single movie
const fetchMovie = async (title) => {
    try {
        const res = await axios.get(BASE_URL, {
            params: {
                apikey: API_KEY,
                t: title
            }
        });
        return res.data;
    } catch (e) {
        console.error(`Error fetching ${title}:`, e);
        return null;
    }
};

// Curated lists since OMDb doesn't have a discover/trending endpoint
const TRENDING_TITLES = ["Barbie", "Oppenheimer", "Dune: Part Two", "The Batman", "Wonka", "Poor Things", "The Marvels", "Aquaman and the Lost Kingdom", "Civil War", "Challengers"];
const TOP_RATED_TITLES = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", "Forrest Gump", "Inception", "Fight Club", "The Matrix", "Goodfellas", "Interstellar"];
const ACTION_TITLES = ["Mad Max: Fury Road", "John Wick: Chapter 4", "Gladiator", "Die Hard", "Top Gun: Maverick", "Mission: Impossible - Dead Reckoning Part One", "The Avengers", "Black Panther", "Logan", "Casino Royale"];
const COMEDY_TITLES = ["Superbad", "The Hangover", "Mean Girls", "Bridesmaids", "Step Brothers", "Anchorman", "Dumb and Dumber", "Ferris Bueller's Day Off"];
const SCIFI_TITLES = ["Blade Runner 2049", "Arrival", "Ex Machina", "Eternal Sunshine of the Spotless Mind", "Her", "2001: A Space Odyssey", "Star Wars: A New Hope", "Alien"];

const getHighResPoster = (url) => {
    if (!url || url === "N/A") return "https://via.placeholder.com/300x450";
    // OMDb returns SX300 by default. Replace with bigger size or remove limit.
    return url.replace("SX300", "SX700");
};

const mapOmdbToApp = (movie, index, category) => {
    if (!movie || movie.Response === "False") return null;

    const highResPoster = getHighResPoster(movie.Poster);

    return {
        id: movie.imdbID,
        title: movie.Title,
        description: movie.Plot,
        poster: highResPoster,
        thumbnail: highResPoster, // Use same high-res for thumbnail
        rating: parseFloat(movie.imdbRating),
        releaseDate: movie.Released,
        videoSrc: getVideo(index),
        category: category
    };
};

export const fetchTrendingMovies = async () => {
    try {
        const promises = TRENDING_TITLES.map(title => fetchMovie(title));
        const results = await Promise.all(promises);

        return results
            .map((movie, index) => mapOmdbToApp(movie, index, "Trending"))
            .filter(Boolean); // Remove failed fetches
    } catch (e) {
        console.error("OMDb Trending Error:", e);
        return [];
    }
};

export const fetchTopRatedMovies = async () => {
    try {
        const promises = TOP_RATED_TITLES.map(title => fetchMovie(title));
        const results = await Promise.all(promises);

        return results
            .map((movie, index) => mapOmdbToApp(movie, index, "Top Rated"))
            .filter(Boolean);
    } catch (e) {
        console.error("OMDb TopRated Error:", e);
        return [];
    }
};

export const fetchActionMovies = async () => {
    try {
        const promises = ACTION_TITLES.map(title => fetchMovie(title));
        const results = await Promise.all(promises);

        return results
            .map((movie, index) => mapOmdbToApp(movie, index, "Action"))
            .filter(Boolean);
    } catch (e) {
        console.error("OMDb Action Error:", e);
        return [];
    }
};

export const fetchComedyMovies = async () => {
    try {
        const promises = COMEDY_TITLES.map(title => fetchMovie(title));
        const results = await Promise.all(promises);

        return results
            .map((movie, index) => mapOmdbToApp(movie, index, "Comedy"))
            .filter(Boolean);
    } catch (e) {
        console.error("OMDb Comedy Error:", e);
        return [];
    }
};

export const fetchSciFiMovies = async () => {
    try {
        const promises = SCIFI_TITLES.map(title => fetchMovie(title));
        const results = await Promise.all(promises);

        return results
            .map((movie, index) => mapOmdbToApp(movie, index, "Sci-Fi"))
            .filter(Boolean);
    } catch (e) {
        console.error("OMDb SciFi Error:", e);
        return [];
    }
};
