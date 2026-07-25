{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}

-- | mao-kernel — the judge/referee decision kernel.
--
-- The Python orchestrator delegates its trust-critical decisions here:
--
--   * @aggregate@     — blind-tournament score aggregation: mean judge
--                       score per candidate, ranking, incumbent-stable
--                       tie-break, winner selection.
--   * @review_edit@   — referee rules over an edit phase: test-file
--                       tampering (reward hacking), metrics-file
--                       fabrication, no-op edits.
--   * @review_result@ — referee rules over a trial result: verdicts
--                       contradicting the numeric ground truth,
--                       premature goal claims, repeated dead ends,
--                       suspicious jumps.
--
-- Protocol: ONE JSON request on stdin -> ONE JSON response on stdout.
-- Every function here is total; unknown ops produce an @error@ field
-- and exit code 2. Behavior must stay identical to the Python fallback
-- in @src/ml_orchestrator/core/{tournament,referee}.py@ — both are
-- checked against @tests/kernel_vectors.json@.
module Main (main) where

import Data.Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.List (isInfixOf, isSuffixOf, sortBy)
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Ord (Down (..), comparing)
import qualified Data.Text as T
import System.Exit (exitWith, ExitCode (..))

main :: IO ()
main = do
  input <- BL.getContents
  case eitherDecode input :: Either String Value of
    Left err -> do
      BL.putStrLn (encode (object ["error" .= ("bad json: " <> err)]))
      exitWith (ExitFailure 2)
    Right req -> case dispatch req of
      Left err -> do
        BL.putStrLn (encode (object ["error" .= err]))
        exitWith (ExitFailure 2)
      Right resp -> BL.putStrLn (encode resp)

dispatch :: Value -> Either String Value
dispatch req = case getString "op" req of
  Just "aggregate"     -> Right (aggregate req)
  Just "review_edit"   -> Right (reviewEdit req)
  Just "review_result" -> Right (reviewResult req)
  Just other           -> Left ("unknown op: " <> T.unpack other)
  Nothing              -> Left "missing op"

-- ---------------------------------------------------------------------------
-- JSON helpers (total accessors over the request object)
-- ---------------------------------------------------------------------------

getField :: T.Text -> Value -> Maybe Value
getField k (Object o) = KM.lookup (Key.fromText k) o
getField _ _ = Nothing

getString :: T.Text -> Value -> Maybe T.Text
getString k v = getField k v >>= \case String s -> Just s; _ -> Nothing

getBool :: T.Text -> Value -> Bool
getBool k v = case getField k v of Just (Bool b) -> b; _ -> False

getNum :: T.Text -> Value -> Maybe Double
getNum k v = getField k v >>= \case Number n -> Just (realToFrac n)
                                    _        -> Nothing

getStrings :: T.Text -> Value -> [T.Text]
getStrings k v = case getField k v of
  Just (Array xs) -> mapMaybe (\case String s -> Just s; _ -> Nothing)
                              (foldr (:) [] xs)
  _ -> []

getStrMap :: T.Text -> Value -> M.Map T.Text T.Text
getStrMap k v = case getField k v of
  Just (Object o) -> M.fromList
    [ (Key.toText kk, s) | (kk, String s) <- KM.toList o ]
  _ -> M.empty

getScoreMap :: T.Text -> Value -> M.Map T.Text (M.Map T.Text Double)
getScoreMap k v = case getField k v of
  Just (Object o) -> M.fromList
    [ (Key.toText kk, inner iv) | (kk, iv) <- KM.toList o ]
  _ -> M.empty
  where
    inner (Object o2) = M.fromList
      [ (Key.toText kk, realToFrac n)
      | (kk, Number n) <- KM.toList o2 ]
    inner _ = M.empty

flag :: T.Text -> T.Text -> T.Text -> Value
flag code sev detail =
  object ["code" .= code, "severity" .= sev, "detail" .= detail]

-- ---------------------------------------------------------------------------
-- aggregate — judge panel aggregation + seat assignment
-- ---------------------------------------------------------------------------

aggregate :: Value -> Value
aggregate req =
  let labels     = getStrings "labels" req
      labelMap   = getStrMap "label_map" req
      judges     = getScoreMap "judge_scores" req
      incumbent  = getString "incumbent" req
      -- mean score per label across all judges that scored it
      meanFor l  = let vals = mapMaybe (M.lookup l) (M.elems judges)
                   in if null vals then Nothing
                      else Just (sum vals / fromIntegral (length vals))
      agentMeans = [ (agent, m)
                   | l <- labels
                   , Just agent <- [M.lookup l labelMap]
                   , Just m <- [meanFor l] ]
      ranking    = sortBy (comparing (Down . snd)) agentMeans
  in case ranking of
       [] -> object ["error" .= ("no scores" :: T.Text)]
       ((_, topScore):_) ->
         let tied   = [a | (a, s) <- ranking, abs (s - topScore) < 1e-9]
             winner = case incumbent of
                        Just inc | inc `elem` tied -> inc
                        _                          -> head tied
             others = [a | (a, _) <- ranking, a /= winner]
             evaluator = fromMaybe winner (safeHead others)
         in object
              [ "ranking"   .= [[toJSON a, toJSON s] | (a, s) <- ranking]
              , "winner"    .= winner
              , "evaluator" .= evaluator
              ]
  where
    safeHead (x:_) = Just x
    safeHead []    = Nothing

-- ---------------------------------------------------------------------------
-- review_edit — referee rules for the edit phase
-- ---------------------------------------------------------------------------

isTestPath :: T.Text -> Bool
isTestPath t =
  let p    = T.unpack t
      base = reverse (takeWhile (/= '/') (reverse p))
      segs = wordsBy '/' p
  in any (`elem` segs) ["test", "tests"]
     || base == "conftest.py"
     || base == "pytest.ini"
     || (("test_" `isInfixOf` base || "tests_" `isInfixOf` base)
         && ".py" `isSuffixOf` base)
     || "_test.py" `isSuffixOf` base
     || "_tests.py" `isSuffixOf` base
     || any (`isSuffixOf` base)
         [".test.js", ".test.ts", ".test.jsx", ".test.tsx",
          ".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx"]
  where
    wordsBy c s = case break (== c) s of
      (w, [])      -> [w]
      (w, _ : rest) -> w : wordsBy c rest

reviewEdit :: Value -> Value
reviewEdit req =
  let mode          = fromMaybe "ml" (getString "mode" req)
      metricsFile   = fromMaybe "metrics.json" (getString "metrics_file" req)
      changed       = getStrings "changed_files" req
      metricsWrote  = getBool "metrics_written" req
      touchedTests  = filter isTestPath changed
      tamperFlag =
        [ flag "TEST_TAMPERING" "CRITICAL"
            ("Editor modified test files in coding mode: "
             <> T.intercalate ", " (take 5 touchedTests)
             <> ". Fitness cannot be trusted; forcing revert.")
        | mode == "coding", not (null touchedTests) ]
      metricsFlag =
        [ flag "METRICS_TAMPERING" "WARN"
            ("Editor wrote '" <> metricsFile <> "' directly during the "
             <> "edit phase (it was deleted before the run, but this is "
             <> "a fabrication attempt signal).")
        | metricsWrote ]
      noopFlag =
        [ flag "NO_CHANGES" "INFO"
            "Editor reported completion without modifying any file."
        | null changed ]
  in object ["flags" .= (tamperFlag <> metricsFlag <> noopFlag)]

-- ---------------------------------------------------------------------------
-- review_result — referee rules for the result phase
-- ---------------------------------------------------------------------------

reviewResult :: Value -> Value
reviewResult req =
  let status    = fromMaybe "" (getString "status" req)
      measured  = getBool "measured" req
      score     = getNum "score" req
      best      = getNum "best_score" req
      goalMet   = getBool "goal_met" req
      isBetter  = getBool "is_better" req
      deadEnd   = getString "repeated_dead_end" req
      direction = fromMaybe "minimize" (getString "direction" req)
      showN     = T.pack . show
      onCrash =
        [ flag "VERDICT_ON_CRASH" "CRITICAL"
            ("Evaluator returned " <> status
             <> " but the run produced no measurable objective.")
        | status `elem` ["IMPROVED", "GOAL_REACHED"], not measured ]
      contradiction =
        [ flag "VERDICT_CONTRADICTION" "WARN"
            ("Evaluator said IMPROVED but score=" <> maybe "null" showN score
             <> " does not beat best=" <> maybe "null" showN best
             <> "; numeric truth wins.")
        | status == "IMPROVED", measured
        , Just _ <- [best], not isBetter ]
      premature =
        [ flag "PREMATURE_GOAL" "WARN"
            ("Evaluator declared GOAL_REACHED but score="
             <> maybe "null" showN score
             <> " does not satisfy the numeric goal condition.")
        | status == "GOAL_REACHED", measured, not goalMet ]
      repeated =
        [ flag "REPEATED_DEAD_END" "WARN"
            ("This trial repeats a change already recorded as a dead end: "
             <> d)
        | Just d <- [deadEnd] ]
      jump = case (measured, score, best) of
        (True, Just s, Just b) | b /= 0 ->
          let j = if direction == "minimize"
                    then (b - s) / abs b else (s - b) / abs b
          in [ flag "SUSPICIOUS_JUMP" "INFO"
                 ("Objective improved by "
                  <> T.pack (show (round (j * 100) :: Integer))
                  <> "% in a single trial — verify it is not an "
                  <> "evaluation artifact.")
             | j > 0.9 ]
        _ -> []
  in object
       ["flags" .= (onCrash <> contradiction <> premature <> repeated <> jump)]
